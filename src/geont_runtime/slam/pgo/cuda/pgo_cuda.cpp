#include <torch/extension.h>

#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <thread>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rotation_blocks_cuda(
    torch::Tensor rotations,
    torch::Tensor meas_rotations,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> translation_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust);

void se3_scale_weighted_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor rel_log_scales,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor scale_sqrt_info,
    double huber_delta,
    torch::Tensor source_block,
    torch::Tensor target_block,
    torch::Tensor edge_residual,
    torch::Tensor robust,
    torch::Tensor summary);

void se3_scale_normal_equations_cuda(
    torch::Tensor source_block,
    torch::Tensor target_block,
    torch::Tensor edge_residual,
    torch::Tensor hessian_group_offsets,
    torch::Tensor hessian_metadata,
    torch::Tensor gradient_group_offsets,
    torch::Tensor gradient_metadata,
    torch::Tensor normal);

namespace {

#define CHECK_FLOAT_OR_DOUBLE(x) \
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble, #x " must be float32 or float64")
#define CHECK_LONG_CPU_OR_CUDA(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")

template<int BLOCK_D>
inline bool fixed_dim_cpu(int node, int col, int anchor) {
    return node == anchor && col < BLOCK_D;
}

inline int64_t sparse_key(int row, int col) {
    return (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
}

template<int RES_D, int BLOCK_D, int POSE_D, bool USE_LDLT = false>
class CachedEigenSimplicialSolver {
public:
    CachedEigenSimplicialSolver(
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
        if constexpr (RES_D == 7 && BLOCK_D == 7 && POSE_D == 6) {
            if (ii.is_cuda()) {
                init_cuda_normal_assembly(ii.device());
            }
        }
    }

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        double damping) {
        check_cached_blocks(source_block, target_block, edge_residual);

        const auto cpu_double = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
        if (can_assemble_on_cuda(source_block)) {
            assemble_cuda_blocks(source_block, target_block, edge_residual);
            return solve_assembled(damping, source_block.options());
        }
        auto source_cpu = source_block.to(cpu_double).contiguous();
        auto target_cpu = target_block.to(cpu_double).contiguous();
        auto residual_cpu = edge_residual.to(cpu_double).contiguous();
        return solve_cpu_blocks(
            source_cpu,
            target_cpu,
            residual_cpu,
            damping,
            source_block.options());
    }

    torch::Tensor solve_multi_rhs(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        torch::Tensor scale_rhs,
        double damping) {
        static_assert(BLOCK_D == 7 || BLOCK_D == 4, "multi-RHS requires a scale state");
        check_cached_blocks(source_block, target_block, edge_residual);
        CHECK_FLOAT_OR_DOUBLE(scale_rhs);
        TORCH_CHECK(
            scale_rhs.dim() == 2 && scale_rhs.size(0) == n_nodes_,
            "scale_rhs must be (N, K)");
        TORCH_CHECK(scale_rhs.size(1) > 0, "scale_rhs must contain at least one mode");

        const auto cpu_double = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
        auto scale_rhs_cpu = scale_rhs.to(cpu_double).contiguous();
        if (can_assemble_on_cuda(source_block)) {
            assemble_cuda_blocks(source_block, target_block, edge_residual);
        } else {
            auto source_cpu = source_block.to(cpu_double).contiguous();
            auto target_cpu = target_block.to(cpu_double).contiguous();
            auto residual_cpu = edge_residual.to(cpu_double).contiguous();
            assemble_cpu_blocks(source_cpu, target_cpu, residual_cpu);
        }
        return solve_assembled_multi_rhs(damping, scale_rhs_cpu, source_block.options());
    }

    torch::Tensor solve_cpu_blocks(
        const torch::Tensor& source_cpu,
        const torch::Tensor& target_cpu,
        const torch::Tensor& residual_cpu,
        double damping,
        const torch::TensorOptions& output_options) {
        assemble_cpu_blocks(source_cpu, target_cpu, residual_cpu);
        return solve_assembled(damping, output_options);
    }

    void assemble_cpu_blocks(
        const torch::Tensor& source_cpu,
        const torch::Tensor& target_cpu,
        const torch::Tensor& residual_cpu) {
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

        for (int node = 0; node < n_nodes_; ++node) {
            for (int col = 0; col < BLOCK_D; ++col) {
                if (fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                    values[diag_value_indices_[node * BLOCK_D + col]] += 1.0;
                }
            }
        }

    }

    torch::Tensor solve_assembled(double damping, const torch::TensorOptions& output_options) {
        double* values = A_.valuePtr();
        for (int node = 0; node < n_nodes_; ++node) {
            for (int col = 0; col < BLOCK_D; ++col) {
                if (!fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                    values[diag_value_indices_[node * BLOCK_D + col]] += damping;
                }
            }
        }

        solver_.factorize(A_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen sparse factorization failed");
        Eigen::VectorXd step = solver_.solve(-grad_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen sparse solve failed");
        for (int idx = 0; idx < step.size(); ++idx) {
            TORCH_CHECK(std::isfinite(step[idx]), "Eigen sparse solver returned a non-finite step");
        }

        if (output_options.dtype() == torch::kFloat32) {
            Eigen::VectorXf step_float = step.cast<float>();
            const auto cpu_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
            auto step_cpu = torch::empty({n_vars_}, cpu_float);
            std::memcpy(
                step_cpu.data_ptr<float>(),
                step_float.data(),
                static_cast<size_t>(n_vars_) * sizeof(float));
            return step_cpu.to(output_options);
        }
        const auto cpu_double = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
        auto step_cpu = torch::empty({n_vars_}, cpu_double);
        std::memcpy(step_cpu.data_ptr<double>(), step.data(), static_cast<size_t>(n_vars_) * sizeof(double));
        return step_cpu.to(output_options);
    }

    torch::Tensor solve_assembled_multi_rhs(
        double damping,
        const torch::Tensor& scale_rhs_cpu,
        const torch::TensorOptions& output_options) {
        double* values = A_.valuePtr();
        for (int node = 0; node < n_nodes_; ++node) {
            for (int col = 0; col < BLOCK_D; ++col) {
                if (!fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                    values[diag_value_indices_[node * BLOCK_D + col]] += damping;
                }
            }
        }

        solver_.factorize(A_);
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen sparse factorization failed");
        Eigen::MatrixXd rhs = make_multi_rhs(scale_rhs_cpu);
        Eigen::MatrixXd solutions(rhs.rows(), rhs.cols());
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(rhs.cols() - 1));
        for (int col = 1; col < rhs.cols(); ++col) {
            workers.emplace_back([this, &rhs, &solutions, col]() {
                solutions.col(col) = solver_.solve(rhs.col(col));
            });
        }
        solutions.col(0) = solver_.solve(rhs.col(0));
        for (auto& worker : workers) {
            worker.join();
        }
        TORCH_CHECK(solutions.allFinite(), "Eigen sparse solver returned non-finite multi-RHS solutions");
        return eigen_matrix_to_tensor(solutions, output_options);
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
    using SparseOrdering = Eigen::NaturalOrdering<int>;
    using SparseSolver = std::conditional_t<
        USE_LDLT,
        Eigen::SimplicialLDLT<SparseMatrix, Eigen::Lower, SparseOrdering>,
        Eigen::SimplicialLLT<SparseMatrix, Eigen::Lower, SparseOrdering>>;

    bool can_assemble_on_cuda(const torch::Tensor& source_block) const {
        if constexpr (RES_D == 7 && BLOCK_D == 7 && POSE_D == 6) {
            return cuda_normal_assembly_ && source_block.is_cuda() && source_block.scalar_type() == at::kFloat;
        }
        return false;
    }

    static torch::Tensor int_vector_to_device(
        const std::vector<int>& values,
        const c10::Device& device) {
        auto cpu = torch::from_blob(
            const_cast<int*>(values.data()),
            {static_cast<int64_t>(values.size())},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt32));
        return cpu.to(device);
    }

    static torch::Tensor int64_vector_to_device(
        const std::vector<int64_t>& values,
        const c10::Device& device) {
        auto cpu = torch::from_blob(
            const_cast<int64_t*>(values.data()),
            {static_cast<int64_t>(values.size())},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64));
        return cpu.to(device);
    }

    void init_cuda_normal_assembly(const c10::Device& device) {
        const int n_values = static_cast<int>(A_.nonZeros());
        std::vector<int> hessian_offsets(static_cast<size_t>(n_values) + 1, 0);
        for (const auto& contribution : hessian_contributions_) {
            ++hessian_offsets[static_cast<size_t>(contribution.value_index) + 1];
        }
        for (int idx = 0; idx < n_values; ++idx) {
            hessian_offsets[idx + 1] += hessian_offsets[idx];
        }
        std::vector<int> hessian_cursor = hessian_offsets;
        std::vector<int64_t> hessian_metadata(hessian_contributions_.size());
        for (const auto& contribution : hessian_contributions_) {
            const int slot = hessian_cursor[contribution.value_index]++;
            const uint64_t edge = static_cast<uint32_t>(
                contribution.edge_offset / (RES_D * BLOCK_D));
            const uint64_t packed =
                edge |
                (static_cast<uint64_t>(contribution.block_a) << 32) |
                (static_cast<uint64_t>(contribution.block_b) << 33) |
                (static_cast<uint64_t>(contribution.col_a) << 34) |
                (static_cast<uint64_t>(contribution.col_b) << 37) |
                (static_cast<uint64_t>(contribution.residual_mask) << 40);
            hessian_metadata[slot] = static_cast<int64_t>(packed);
        }

        std::vector<int> gradient_offsets(static_cast<size_t>(n_vars_) + 1, 0);
        for (int e = 0; e < n_edges_; ++e) {
            const int nodes[2] = {ii_[e], jj_[e]};
            for (int block_kind = 0; block_kind < 2; ++block_kind) {
                const int node = nodes[block_kind];
                for (int col = 0; col < BLOCK_D; ++col) {
                    if (fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                        continue;
                    }
                    if (residual_support_mask(block_kind, col) != 0) {
                        ++gradient_offsets[static_cast<size_t>(node * BLOCK_D + col) + 1];
                    }
                }
            }
        }
        for (int idx = 0; idx < n_vars_; ++idx) {
            gradient_offsets[idx + 1] += gradient_offsets[idx];
        }
        std::vector<int> gradient_cursor = gradient_offsets;
        std::vector<int64_t> gradient_metadata(static_cast<size_t>(gradient_offsets.back()));
        for (int e = 0; e < n_edges_; ++e) {
            const int nodes[2] = {ii_[e], jj_[e]};
            for (int block_kind = 0; block_kind < 2; ++block_kind) {
                const int node = nodes[block_kind];
                for (int col = 0; col < BLOCK_D; ++col) {
                    if (fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                        continue;
                    }
                    const int residual_mask = residual_support_mask(block_kind, col);
                    if (residual_mask == 0) {
                        continue;
                    }
                    const int variable = node * BLOCK_D + col;
                    const int slot = gradient_cursor[variable]++;
                    const uint64_t packed =
                        static_cast<uint32_t>(e) |
                        (static_cast<uint64_t>(block_kind) << 32) |
                        (static_cast<uint64_t>(col) << 33) |
                        (static_cast<uint64_t>(residual_mask) << 36);
                    gradient_metadata[slot] = static_cast<int64_t>(packed);
                }
            }
        }

        hessian_group_offsets_cuda_ = int_vector_to_device(hessian_offsets, device);
        hessian_metadata_cuda_ = int64_vector_to_device(hessian_metadata, device);
        gradient_group_offsets_cuda_ = int_vector_to_device(gradient_offsets, device);
        gradient_metadata_cuda_ = int64_vector_to_device(gradient_metadata, device);
        const int64_t normal_size = static_cast<int64_t>(A_.nonZeros()) + n_vars_;
        normal_cuda_ = torch::empty(
            {normal_size},
            torch::TensorOptions().device(device).dtype(torch::kFloat64));
        normal_cpu_ = torch::empty(
            {normal_size},
            torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64).pinned_memory(true));
        hessian_contributions_.clear();
        hessian_contributions_.shrink_to_fit();
        cuda_normal_assembly_ = true;
    }

    void assemble_cuda_blocks(
        const torch::Tensor& source_block,
        const torch::Tensor& target_block,
        const torch::Tensor& residual) {
        se3_scale_normal_equations_cuda(
            source_block,
            target_block,
            residual,
            hessian_group_offsets_cuda_,
            hessian_metadata_cuda_,
            gradient_group_offsets_cuda_,
            gradient_metadata_cuda_,
            normal_cuda_);
        normal_cpu_.copy_(normal_cuda_);
        TORCH_CHECK(
            normal_cpu_.numel() == A_.nonZeros() + n_vars_,
            "CUDA normal-equation buffer size mismatch");
        const double* normal_ptr = normal_cpu_.data_ptr<double>();
        std::memcpy(
            A_.valuePtr(),
            normal_ptr,
            static_cast<size_t>(A_.nonZeros()) * sizeof(double));
        std::memcpy(
            grad_.data(),
            normal_ptr + A_.nonZeros(),
            static_cast<size_t>(n_vars_) * sizeof(double));

        double* values = A_.valuePtr();
        for (int node = 0; node < n_nodes_; ++node) {
            for (int col = 0; col < BLOCK_D; ++col) {
                if (fixed_dim_cpu<BLOCK_D>(node, col, anchor_)) {
                    values[diag_value_indices_[node * BLOCK_D + col]] += 1.0;
                }
            }
        }
    }

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
        if constexpr (RES_D == 7 && BLOCK_D == 7 && POSE_D == 6) {
            constexpr uint8_t translation_rows = 0x07;
            constexpr uint8_t pose_rows = 0x3f;
            constexpr uint8_t scale_row = 0x40;
            if (block_kind == 0) {
                if (col < 3) {
                    return translation_rows;
                }
                if (col < 6) {
                    return pose_rows;
                }
                return translation_rows | scale_row;
            }
            if (col < 3) {
                return translation_rows;
            }
            if (col < 6) {
                return pose_rows;
            }
            return scale_row;
        }
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
        } else if constexpr (RES_D == 7) {
            double value = 0.0;
            if ((residual_mask & 0x07) != 0) {
                value +=
                    block[col] * residual[0] +
                    block[BLOCK_D + col] * residual[1] +
                    block[2 * BLOCK_D + col] * residual[2];
            }
            if ((residual_mask & 0x38) != 0) {
                value +=
                    block[3 * BLOCK_D + col] * residual[3] +
                    block[4 * BLOCK_D + col] * residual[4] +
                    block[5 * BLOCK_D + col] * residual[5];
            }
            if ((residual_mask & 0x40) != 0) {
                value += block[6 * BLOCK_D + col] * residual[6];
            }
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
        } else if constexpr (RES_D == 7) {
            double value = 0.0;
            if ((residual_mask & 0x07) != 0) {
                value +=
                    block_a[col_a] * block_b[col_b] +
                    block_a[BLOCK_D + col_a] * block_b[BLOCK_D + col_b] +
                    block_a[2 * BLOCK_D + col_a] * block_b[2 * BLOCK_D + col_b];
            }
            if ((residual_mask & 0x38) != 0) {
                value +=
                    block_a[3 * BLOCK_D + col_a] * block_b[3 * BLOCK_D + col_b] +
                    block_a[4 * BLOCK_D + col_a] * block_b[4 * BLOCK_D + col_b] +
                    block_a[5 * BLOCK_D + col_a] * block_b[5 * BLOCK_D + col_b];
            }
            if ((residual_mask & 0x40) != 0) {
                value += block_a[6 * BLOCK_D + col_a] * block_b[6 * BLOCK_D + col_b];
            }
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
            if (fixed_dim_cpu<BLOCK_D>(node, c, anchor_)) {
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
            if (fixed_dim_cpu<BLOCK_D>(node_a, a, anchor_)) {
                continue;
            }
            const uint8_t residual_mask_a = residual_support_mask(block_a, a);
            if (residual_mask_a == 0) {
                continue;
            }
            const int row_index = node_a * BLOCK_D + a;
            for (int b = 0; b < BLOCK_D; ++b) {
                if (fixed_dim_cpu<BLOCK_D>(node_b, b, anchor_)) {
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
            if (fixed_dim_cpu<BLOCK_D>(node_a, a, anchor_)) {
                continue;
            }
            const int row_index = node_a * BLOCK_D + a;
            for (int b = 0; b < BLOCK_D; ++b) {
                if (fixed_dim_cpu<BLOCK_D>(node_b, b, anchor_)) {
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

    Eigen::MatrixXd make_multi_rhs(const torch::Tensor& scale_rhs_cpu) const {
        const int n_modes = static_cast<int>(scale_rhs_cpu.size(1));
        Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(n_vars_, n_modes + 1);
        rhs.col(0) = -grad_;
        const double* scale_ptr = scale_rhs_cpu.data_ptr<double>();
        for (int node = 0; node < n_nodes_; ++node) {
            if (node == anchor_) {
                continue;
            }
            for (int mode = 0; mode < n_modes; ++mode) {
                rhs(node * BLOCK_D + (BLOCK_D - 1), mode + 1) =
                    scale_ptr[node * n_modes + mode];
            }
        }
        return rhs;
    }

    torch::Tensor eigen_matrix_to_tensor(
        const Eigen::MatrixXd& matrix,
        const torch::TensorOptions& output_options) const {
        if (output_options.dtype() == torch::kFloat32) {
            using RowMajorFloatMatrix =
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
            RowMajorFloatMatrix row_major = matrix.cast<float>();
            const auto cpu_float = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32);
            auto output_cpu = torch::empty({matrix.rows(), matrix.cols()}, cpu_float);
            std::memcpy(
                output_cpu.data_ptr<float>(),
                row_major.data(),
                static_cast<size_t>(matrix.size()) * sizeof(float));
            return output_cpu.to(output_options);
        }
        using RowMajorMatrix =
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        RowMajorMatrix row_major = matrix;
        const auto cpu_double = torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat64);
        auto output_cpu = torch::empty({matrix.rows(), matrix.cols()}, cpu_double);
        std::memcpy(
            output_cpu.data_ptr<double>(),
            row_major.data(),
            static_cast<size_t>(matrix.size()) * sizeof(double));
        return output_cpu.to(output_options);
    }

    void build_pattern() {
        std::vector<Eigen::Triplet<double>> triplets;
        std::unordered_set<uint64_t> block_patterns;
        const auto block_key = [](int node_a, int node_b, uint8_t block_a, uint8_t block_b) {
            return (static_cast<uint64_t>(static_cast<uint32_t>(node_a)) << 33) |
                   (static_cast<uint64_t>(static_cast<uint32_t>(node_b)) << 2) |
                   (static_cast<uint64_t>(block_a) << 1) |
                   static_cast<uint64_t>(block_b);
        };
        block_patterns.reserve(static_cast<size_t>(n_edges_) + static_cast<size_t>(n_nodes_) * 2);
        for (int e = 0; e < n_edges_; ++e) {
            const int i = ii_[e];
            const int j = jj_[e];
            block_patterns.insert(block_key(i, i, 0, 0));
            block_patterns.insert(block_key(j, j, 1, 1));
            if (i >= j) {
                block_patterns.insert(block_key(i, j, 0, 1));
            }
            if (j >= i) {
                block_patterns.insert(block_key(j, i, 1, 0));
            }
        }
        triplets.reserve(block_patterns.size() * BLOCK_D * BLOCK_D + static_cast<size_t>(n_vars_));
        for (const uint64_t key : block_patterns) {
            const int node_a = static_cast<int>(key >> 33);
            const int node_b = static_cast<int>((key >> 2) & 0x7fffffffu);
            const uint8_t block_a = static_cast<uint8_t>((key >> 1) & 1u);
            const uint8_t block_b = static_cast<uint8_t>(key & 1u);
            add_pair_pattern(triplets, node_a, node_b, block_a, block_b);
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
        TORCH_CHECK(solver_.info() == Eigen::Success, "Eigen sparse analyzePattern failed");
    }

    int n_nodes_;
    int anchor_;
    int n_edges_;
    int n_vars_;
    bool cuda_normal_assembly_ = false;
    std::vector<int> ii_;
    std::vector<int> jj_;
    std::vector<int> diag_value_indices_;
    std::vector<HessianContribution> hessian_contributions_;
    torch::Tensor hessian_group_offsets_cuda_;
    torch::Tensor hessian_metadata_cuda_;
    torch::Tensor gradient_group_offsets_cuda_;
    torch::Tensor gradient_metadata_cuda_;
    torch::Tensor normal_cuda_;
    torch::Tensor normal_cpu_;
    SparseMatrix A_;
    Eigen::VectorXd grad_;
    SparseSolver solver_;
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
        return solver_.solve(source_block, target_block, edge_residual, damping);
    }

private:
    CachedEigenSimplicialSolver<3, 3, 3> solver_;
};

class TranslationScaleEigenSimplicialLLTSolver {
public:
    TranslationScaleEigenSimplicialLLTSolver(torch::Tensor ii, torch::Tensor jj, int64_t n_nodes, int64_t anchor)
        : solver_(ii, jj, n_nodes, anchor) {}

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        double damping) {
        return solver_.solve(source_block, target_block, edge_residual, damping);
    }

private:
    CachedEigenSimplicialSolver<4, 4, 3> solver_;
};

class Se3ScaleEigenSimplicialLDLTSolver {
public:
    Se3ScaleEigenSimplicialLDLTSolver(torch::Tensor ii, torch::Tensor jj, int64_t n_nodes, int64_t anchor)
        : solver_(ii, jj, n_nodes, anchor) {}

    torch::Tensor solve(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        double damping) {
        return solver_.solve(source_block, target_block, edge_residual, damping);
    }

    torch::Tensor solve_multi_rhs(
        torch::Tensor source_block,
        torch::Tensor target_block,
        torch::Tensor edge_residual,
        torch::Tensor scale_rhs,
        double damping) {
        return solver_.solve_multi_rhs(
            source_block,
            target_block,
            edge_residual,
            scale_rhs,
            damping);
    }

private:
    CachedEigenSimplicialSolver<7, 7, 6, true> solver_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rotation_blocks", &rotation_blocks_cuda, "rotation PGO CUDA block assembly");
    m.def("translation_scale_blocks", &translation_scale_blocks_cuda, "translation+scale PGO CUDA block assembly");
    m.def("se3_scale_blocks", &se3_scale_blocks_cuda, "SE3+scale PGO CUDA block assembly");
    m.def("se3_scale_weighted_blocks", &se3_scale_weighted_blocks_cuda, "SE3+scale PGO CUDA weighted block assembly");
    pybind11::class_<RotationEigenSimplicialLLTSolver>(m, "RotationEigenSimplicialLLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &RotationEigenSimplicialLLTSolver::solve);
    pybind11::class_<TranslationScaleEigenSimplicialLLTSolver>(m, "TranslationScaleEigenSimplicialLLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &TranslationScaleEigenSimplicialLLTSolver::solve);
    pybind11::class_<Se3ScaleEigenSimplicialLDLTSolver>(m, "Se3ScaleEigenSimplicialLDLTSolver")
        .def(pybind11::init<torch::Tensor, torch::Tensor, int64_t, int64_t>())
        .def("solve", &Se3ScaleEigenSimplicialLDLTSolver::solve)
        .def("solve_multi_rhs", &Se3ScaleEigenSimplicialLDLTSolver::solve_multi_rhs);
}
