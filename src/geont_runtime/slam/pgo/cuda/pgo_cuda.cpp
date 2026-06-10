#include <torch/extension.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rotation_blocks_cuda(
    torch::Tensor rotations,
    torch::Tensor meas_rotations,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> translation_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust,
    double scale_prior_diag);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> se3_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust,
    double scale_prior_diag);

namespace {

#define CHECK_FLOAT_OR_DOUBLE(x) \
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, #x " must be float32 or float64")
#define CHECK_LONG_CPU_OR_CUDA(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

template<int BLOCK_D, int POSE_D>
inline bool fixed_dim_cpu(int node, int col, int anchor) {
    return node == anchor && col < POSE_D;
}

inline int64_t sparse_key(int row, int col) {
    return (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
}

template<int RES_D, int BLOCK_D, int POSE_D>
class CachedEigenSimplicialLLT {
public:
    CachedEigenSimplicialLLT(
        torch::Tensor ii,
        torch::Tensor jj,
        int64_t n_nodes,
        int64_t anchor)
        : n_nodes_(static_cast<int>(n_nodes)),
          anchor_(static_cast<int>(anchor)),
          n_vars_(static_cast<int>(n_nodes * BLOCK_D)) {
        CHECK_LONG_CPU_OR_CUDA(ii);
        CHECK_LONG_CPU_OR_CUDA(jj);
        TORCH_CHECK(ii.dim() == 1 && jj.dim() == 1, "ii/jj must be 1-D");
        TORCH_CHECK(ii.numel() == jj.numel(), "ii/jj edge count mismatch");
        TORCH_CHECK(n_nodes > 0, "n_nodes must be positive");
        TORCH_CHECK(anchor >= 0 && anchor < n_nodes, "anchor out of range");

        auto ii_cpu = ii.to(torch::kCPU).contiguous();
        auto jj_cpu = jj.to(torch::kCPU).contiguous();
        const auto n_edges64 = ii_cpu.numel();
        TORCH_CHECK(n_edges64 <= std::numeric_limits<int>::max(), "too many edges for Eigen cached solver");
        n_edges_ = static_cast<int>(n_edges64);
        ii_.resize(n_edges_);
        jj_.resize(n_edges_);
        const int64_t* ii_ptr = ii_cpu.data_ptr<int64_t>();
        const int64_t* jj_ptr = jj_cpu.data_ptr<int64_t>();
        for (int e = 0; e < n_edges_; ++e) {
            const int i = static_cast<int>(ii_ptr[e]);
            const int j = static_cast<int>(jj_ptr[e]);
            TORCH_CHECK(i >= 0 && i < n_nodes_ && j >= 0 && j < n_nodes_, "edge index out of range");
            ii_[e] = i;
            jj_[e] = j;
        }
        build_pattern();
    }

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        torch::Tensor prior_gradient,
        double damping,
        double scale_prior_diag) {
        check_cached_blocks(source_block, target_block, edge_residual);
        if constexpr (BLOCK_D == 7 || BLOCK_D == 4) {
            CHECK_FLOAT_OR_DOUBLE(prior_gradient);
            TORCH_CHECK(prior_gradient.dim() == 1 && prior_gradient.numel() == n_nodes_, "prior_gradient must be (N,)");
        }

        const auto cpu_double = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
        auto source_cpu = source_block.to(cpu_double).contiguous();
        auto target_cpu = target_block.to(cpu_double).contiguous();
        auto residual_cpu = edge_residual.to(cpu_double).contiguous();
        torch::Tensor prior_cpu;
        if constexpr (BLOCK_D == 7 || BLOCK_D == 4) {
            prior_cpu = prior_gradient.to(cpu_double).contiguous();
        }

        std::fill(A_.valuePtr(), A_.valuePtr() + A_.nonZeros(), 0.0);
        grad_.setZero();

        const double* source_ptr = source_cpu.data_ptr<double>();
        const double* target_ptr = target_cpu.data_ptr<double>();
        const double* residual_ptr = residual_cpu.data_ptr<double>();

        for (int e = 0; e < n_edges_; ++e) {
            const double* src = source_ptr + e * RES_D * BLOCK_D;
            const double* tgt = target_ptr + e * RES_D * BLOCK_D;
            const double* res = residual_ptr + e * RES_D;
            add_gradient(src, res, ii_[e], 0);
            add_gradient(tgt, res, jj_[e], 1);
        }

        double* values = A_.valuePtr();
        const double* block_ptrs[2] = {source_ptr, target_ptr};
        for (const auto& contribution : hessian_contributions_) {
            const double* base_a = block_ptrs[contribution.block_a] + contribution.edge_offset;
            const double* base_b = block_ptrs[contribution.block_b] + contribution.edge_offset;
            values[contribution.value_index] += dot_block_columns(
                base_a,
                base_b,
                contribution.col_a,
                contribution.col_b,
                contribution.residual_mask);
        }

        if constexpr (BLOCK_D == 7 || BLOCK_D == 4) {
            const double* prior_ptr = prior_cpu.data_ptr<double>();
            for (int node = 0; node < n_nodes_; ++node) {
                grad_[node * BLOCK_D + (BLOCK_D - 1)] += prior_ptr[node];
            }
        }

        for (int node = 0; node < n_nodes_; ++node) {
            for (int col = 0; col < BLOCK_D; ++col) {
                double diag = damping;
                if (fixed_dim_cpu<BLOCK_D, POSE_D>(node, col, anchor_)) {
                    diag = 1.0;
                } else if constexpr (BLOCK_D == 7 || BLOCK_D == 4) {
                    if (col == BLOCK_D - 1) {
                        diag += scale_prior_diag;
                    }
                }
                values[diag_value_indices_[node * BLOCK_D + col]] += diag;
            }
        }

        solver_.factorize(A_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen SimplicialLLT factorization failed");
        Eigen::VectorXd step = solver_.solve(-grad_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen SimplicialLLT solve failed");
        for (int idx = 0; idx < step.size(); ++idx) {
            TORCH_CHECK(std::isfinite(step[idx]), "Eigen SimplicialLLT returned a non-finite step");
        }

        auto step_cpu = torch::empty({n_vars_}, cpu_double);
        std::memcpy(step_cpu.data_ptr<double>(), step.data(), static_cast<size_t>(n_vars_) * sizeof(double));
        return step_cpu.to(source_block.options());
    }

private:
    struct HessianContribution {
        int edge_offset;
        int value_index;
        uint8_t block_a;
        uint8_t block_b;
        uint8_t col_a;
        uint8_t col_b;
        uint8_t residual_mask;
    };

    using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor>;

    void check_cached_blocks(
        const torch::Tensor& source_block,
        const torch::Tensor& target_block,
        const torch::Tensor& edge_residual) const {
        CHECK_FLOAT_OR_DOUBLE(source_block);
        CHECK_FLOAT_OR_DOUBLE(target_block);
        CHECK_FLOAT_OR_DOUBLE(edge_residual);
        TORCH_CHECK(source_block.dim() == 3, "source_block must be 3-D");
        TORCH_CHECK(target_block.sizes() == source_block.sizes(), "target_block shape mismatch");
        TORCH_CHECK(source_block.size(0) == n_edges_, "source_block edge count mismatch");
        TORCH_CHECK(source_block.size(1) == RES_D, "bad source_block residual dim");
        TORCH_CHECK(source_block.size(2) == BLOCK_D, "bad source_block block dim");
        TORCH_CHECK(edge_residual.dim() == 2, "edge_residual must be 2-D");
        TORCH_CHECK(edge_residual.size(0) == n_edges_, "edge_residual edge count mismatch");
        TORCH_CHECK(edge_residual.size(1) == RES_D, "bad edge_residual dim");
    }

    static constexpr uint8_t all_residual_rows() {
        return static_cast<uint8_t>((1u << RES_D) - 1u);
    }

    static uint8_t residual_support_mask(uint8_t block_kind, int col) {
        if constexpr (RES_D == 6 && BLOCK_D == 7 && POSE_D == 6) {
            constexpr uint8_t translation_rows = 0x07;
            constexpr uint8_t all_rows = 0x3f;
            if (block_kind == 0) {
                return (col < 3 || col == 6) ? translation_rows : all_rows;
            }
            if (col < 3) {
                return translation_rows;
            }
            if (col < 6) {
                return all_rows;
            }
            return 0;
        }
        if constexpr (RES_D == 3 && BLOCK_D == 4 && POSE_D == 3) {
            if (block_kind == 1 && col == 3) {
                return 0;
            }
        }
        return all_residual_rows();
    }

    static double dot_block_residual(
        const double* block,
        const double* residual,
        int col,
        uint8_t residual_mask) {
        if constexpr (RES_D == 3) {
            return block[col] * residual[0] +
                   block[BLOCK_D + col] * residual[1] +
                   block[2 * BLOCK_D + col] * residual[2];
        } else if constexpr (RES_D == 6) {
            double value =
                block[col] * residual[0] +
                block[BLOCK_D + col] * residual[1] +
                block[2 * BLOCK_D + col] * residual[2];
            if (residual_mask == 0x07) {
                return value;
            }
            value +=
                block[3 * BLOCK_D + col] * residual[3] +
                block[4 * BLOCK_D + col] * residual[4] +
                block[5 * BLOCK_D + col] * residual[5];
            return value;
        } else {
            double value = 0.0;
            for (int row = 0; row < RES_D; ++row) {
                if ((residual_mask & (1u << row)) != 0) {
                    value += block[row * BLOCK_D + col] * residual[row];
                }
            }
            return value;
        }
    }

    static double dot_block_columns(
        const double* block_a,
        const double* block_b,
        int col_a,
        int col_b,
        uint8_t residual_mask) {
        if constexpr (RES_D == 3) {
            return block_a[col_a] * block_b[col_b] +
                   block_a[BLOCK_D + col_a] * block_b[BLOCK_D + col_b] +
                   block_a[2 * BLOCK_D + col_a] * block_b[2 * BLOCK_D + col_b];
        } else if constexpr (RES_D == 6) {
            double value =
                block_a[col_a] * block_b[col_b] +
                block_a[BLOCK_D + col_a] * block_b[BLOCK_D + col_b] +
                block_a[2 * BLOCK_D + col_a] * block_b[2 * BLOCK_D + col_b];
            if (residual_mask == 0x07) {
                return value;
            }
            value +=
                block_a[3 * BLOCK_D + col_a] * block_b[3 * BLOCK_D + col_b] +
                block_a[4 * BLOCK_D + col_a] * block_b[4 * BLOCK_D + col_b] +
                block_a[5 * BLOCK_D + col_a] * block_b[5 * BLOCK_D + col_b];
            return value;
        } else {
            double value = 0.0;
            for (int row = 0; row < RES_D; ++row) {
                if ((residual_mask & (1u << row)) != 0) {
                    value += block_a[row * BLOCK_D + col_a] * block_b[row * BLOCK_D + col_b];
                }
            }
            return value;
        }
    }

    void add_gradient(const double* block, const double* residual, int node, uint8_t block_kind) {
        for (int c = 0; c < BLOCK_D; ++c) {
            if (fixed_dim_cpu<BLOCK_D, POSE_D>(node, c, anchor_)) {
                continue;
            }
            const uint8_t residual_mask = residual_support_mask(block_kind, c);
            if (residual_mask == 0) {
                continue;
            }
            grad_[node * BLOCK_D + c] += dot_block_residual(block, residual, c, residual_mask);
        }
    }

    void add_pair_pattern(
        std::vector<Eigen::Triplet<double>>& triplets,
        int node_a,
        int node_b,
        uint8_t block_a,
        uint8_t block_b) const {
        for (int a = 0; a < BLOCK_D; ++a) {
            if (fixed_dim_cpu<BLOCK_D, POSE_D>(node_a, a, anchor_)) {
                continue;
            }
            const uint8_t residual_mask_a = residual_support_mask(block_a, a);
            if (residual_mask_a == 0) {
                continue;
            }
            const int row_index = node_a * BLOCK_D + a;
            for (int b = 0; b < BLOCK_D; ++b) {
                if (fixed_dim_cpu<BLOCK_D, POSE_D>(node_b, b, anchor_)) {
                    continue;
                }
                const uint8_t residual_mask = residual_mask_a & residual_support_mask(block_b, b);
                if (residual_mask == 0) {
                    continue;
                }
                const int col_index = node_b * BLOCK_D + b;
                if (row_index < col_index) {
                    continue;
                }
                triplets.emplace_back(row_index, col_index, 1.0);
            }
        }
    }

    void add_pair_contributions(
        const std::unordered_map<int64_t, int>& value_index_by_coord,
        int edge,
        int node_a,
        int node_b,
        uint8_t block_a,
        uint8_t block_b) {
        for (int a = 0; a < BLOCK_D; ++a) {
            if (fixed_dim_cpu<BLOCK_D, POSE_D>(node_a, a, anchor_)) {
                continue;
            }
            const int row_index = node_a * BLOCK_D + a;
            for (int b = 0; b < BLOCK_D; ++b) {
                if (fixed_dim_cpu<BLOCK_D, POSE_D>(node_b, b, anchor_)) {
                    continue;
                }
                const int col_index = node_b * BLOCK_D + b;
                if (row_index < col_index) {
                    continue;
                }
                const uint8_t residual_mask =
                    residual_support_mask(block_a, a) & residual_support_mask(block_b, b);
                if (residual_mask == 0) {
                    continue;
                }
                const auto found = value_index_by_coord.find(sparse_key(row_index, col_index));
                TORCH_CHECK(found != value_index_by_coord.end(), "cached Eigen sparsity index missing");
                hessian_contributions_.push_back(
                    {edge * RES_D * BLOCK_D,
                     found->second,
                     block_a,
                     block_b,
                     static_cast<uint8_t>(a),
                     static_cast<uint8_t>(b),
                     residual_mask});
            }
        }
    }

    void build_pattern() {
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(static_cast<size_t>(n_edges_) * BLOCK_D * BLOCK_D * 2 + static_cast<size_t>(n_vars_));
        for (int e = 0; e < n_edges_; ++e) {
            const int i = ii_[e];
            const int j = jj_[e];
            add_pair_pattern(triplets, i, i, 0, 0);
            add_pair_pattern(triplets, i, j, 0, 1);
            add_pair_pattern(triplets, j, i, 1, 0);
            add_pair_pattern(triplets, j, j, 1, 1);
        }
        for (int idx = 0; idx < n_vars_; ++idx) {
            triplets.emplace_back(idx, idx, 1.0);
        }

        A_.resize(n_vars_, n_vars_);
        A_.setFromTriplets(triplets.begin(), triplets.end());
        A_.makeCompressed();

        std::unordered_map<int64_t, int> value_index_by_coord;
        value_index_by_coord.reserve(static_cast<size_t>(A_.nonZeros()));
        for (int col = 0; col < A_.outerSize(); ++col) {
            for (SparseMatrix::InnerIterator it(A_, col); it; ++it) {
                const int value_index = static_cast<int>(&it.valueRef() - A_.valuePtr());
                value_index_by_coord.emplace(sparse_key(static_cast<int>(it.row()), static_cast<int>(it.col())), value_index);
            }
        }

        diag_value_indices_.resize(n_vars_);
        for (int idx = 0; idx < n_vars_; ++idx) {
            const auto found = value_index_by_coord.find(sparse_key(idx, idx));
            TORCH_CHECK(found != value_index_by_coord.end(), "cached Eigen diagonal index missing");
            diag_value_indices_[idx] = found->second;
        }

        hessian_contributions_.reserve(static_cast<size_t>(n_edges_) * BLOCK_D * BLOCK_D * 2);
        for (int e = 0; e < n_edges_; ++e) {
            const int i = ii_[e];
            const int j = jj_[e];
            add_pair_contributions(value_index_by_coord, e, i, i, 0, 0);
            add_pair_contributions(value_index_by_coord, e, i, j, 0, 1);
            add_pair_contributions(value_index_by_coord, e, j, i, 1, 0);
            add_pair_contributions(value_index_by_coord, e, j, j, 1, 1);
        }

        grad_ = Eigen::VectorXd::Zero(n_vars_);
        solver_.analyzePattern(A_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen SimplicialLLT analyzePattern failed");
    }

    int n_nodes_;
    int anchor_;
    int n_edges_;
    int n_vars_;
    std::vector<int> ii_;
    std::vector<int> jj_;
    std::vector<int> diag_value_indices_;
    std::vector<HessianContribution> hessian_contributions_;
    SparseMatrix A_;
    Eigen::VectorXd grad_;
    Eigen::SimplicialLLT<SparseMatrix, Eigen::Lower> solver_;
};

class RotationEigenSimplicialLLTSolver {
public:
    RotationEigenSimplicialLLTSolver(torch::Tensor ii, torch::Tensor jj, int64_t n_nodes, int64_t anchor)
        : solver_(ii, jj, n_nodes, anchor) {}

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        double damping) {
        auto prior = torch::empty({0}, source_block.options());
        return solver_.solve(source_block, target_block, edge_residual, prior, damping, 0.0);
    }

private:
    CachedEigenSimplicialLLT<3, 3, 3> solver_;
};

class TranslationScaleEigenSimplicialLLTSolver {
public:
    TranslationScaleEigenSimplicialLLTSolver(torch::Tensor ii, torch::Tensor jj, int64_t n_nodes, int64_t anchor)
        : solver_(ii, jj, n_nodes, anchor) {}

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        torch::Tensor prior_gradient,
        double damping,
        double scale_prior_diag) {
        return solver_.solve(
            source_block,
            target_block,
            edge_residual,
            prior_gradient,
            damping,
            scale_prior_diag);
    }

private:
    CachedEigenSimplicialLLT<3, 4, 3> solver_;
};

class Se3ScaleEigenSimplicialLLTSolver {
public:
    Se3ScaleEigenSimplicialLLTSolver(torch::Tensor ii, torch::Tensor jj, int64_t n_nodes, int64_t anchor)
        : solver_(ii, jj, n_nodes, anchor) {}

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        torch::Tensor prior_gradient,
        double damping,
        double scale_prior_diag) {
        return solver_.solve(
            source_block,
            target_block,
            edge_residual,
            prior_gradient,
            damping,
            scale_prior_diag);
    }

private:
    CachedEigenSimplicialLLT<6, 7, 6> solver_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotation_blocks", &rotation_blocks_cuda, "rotation PGO CUDA block assembly");
    m.def("translation_scale_blocks", &translation_scale_blocks_cuda, "translation+scale PGO CUDA block assembly");
    m.def("se3_scale_blocks", &se3_scale_blocks_cuda, "SE3+scale PGO CUDA block assembly");
    pybind11::class_<RotationEigenSimplicialLLTSolver>(m, "RotationEigenSimplicialLLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &RotationEigenSimplicialLLTSolver::solve);
    pybind11::class_<TranslationScaleEigenSimplicialLLTSolver>(m, "TranslationScaleEigenSimplicialLLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &TranslationScaleEigenSimplicialLLTSolver::solve);
    pybind11::class_<Se3ScaleEigenSimplicialLLTSolver>(m, "Se3ScaleEigenSimplicialLLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &Se3ScaleEigenSimplicialLLTSolver::solve);
}
