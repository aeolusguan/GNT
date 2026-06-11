#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <tuple>

namespace {

constexpr int THREADS = 256;
constexpr float EPS = 1.0e-8f;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_LONG(x) TORCH_CHECK(x.scalar_type() == at::kLong, #x " must be int64")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

void check_edge_index_tensors(const torch::Tensor& ii, const torch::Tensor& jj, int64_t n_edges) {
    CHECK_CUDA(ii);
    CHECK_CUDA(jj);
    CHECK_LONG(ii);
    CHECK_LONG(jj);
    CHECK_CONTIGUOUS(ii);
    CHECK_CONTIGUOUS(jj);
    TORCH_CHECK(ii.dim() == 1 && jj.dim() == 1, "ii/jj must be 1-D");
    TORCH_CHECK(ii.numel() == n_edges && jj.numel() == n_edges, "ii/jj edge count mismatch");
}

void check_weights(const torch::Tensor& sqrt_info, const torch::Tensor& robust, int64_t n_edges, int residual_dim) {
    CHECK_CUDA(sqrt_info);
    CHECK_CUDA(robust);
    CHECK_FLOAT(sqrt_info);
    CHECK_FLOAT(robust);
    CHECK_CONTIGUOUS(sqrt_info);
    CHECK_CONTIGUOUS(robust);
    TORCH_CHECK(sqrt_info.dim() == 2, "sqrt_info must be 2-D");
    TORCH_CHECK(sqrt_info.size(0) == n_edges && sqrt_info.size(1) == residual_dim, "sqrt_info shape mismatch");
    TORCH_CHECK(robust.dim() == 2, "robust must be 2-D");
    TORCH_CHECK(robust.size(0) == n_edges && robust.size(1) == 1, "robust must have shape (E, 1)");
}

__device__ inline float clamp_scale(float value) {
    return value < EPS ? EPS : value;
}

__device__ inline void atomic_max_float(float* address, float value) {
    int* address_as_int = reinterpret_cast<int*>(address);
    int old = *address_as_int;
    while (value > __int_as_float(old)) {
        const int assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
        if (old == assumed) {
            break;
        }
    }
}

__device__ inline void load_quat(const float* data, float q[4]) {
    q[0] = data[0];
    q[1] = data[1];
    q[2] = data[2];
    q[3] = data[3];
}

__device__ inline void normalize_quat(float q[4]) {
    const float norm_sq = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
    const float inv_norm = rsqrtf(fmaxf(norm_sq, EPS));
    q[0] *= inv_norm;
    q[1] *= inv_norm;
    q[2] *= inv_norm;
    q[3] *= inv_norm;
}

__device__ inline void quat_inv(const float q[4], float out[4]) {
    const float norm_sq = fmaxf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3], EPS);
    out[0] = -q[0] / norm_sq;
    out[1] = -q[1] / norm_sq;
    out[2] = -q[2] / norm_sq;
    out[3] = q[3] / norm_sq;
}

__device__ inline void quat_mul(const float a[4], const float b[4], float out[4]) {
    const float ax = a[0];
    const float ay = a[1];
    const float az = a[2];
    const float aw = a[3];
    const float bx = b[0];
    const float by = b[1];
    const float bz = b[2];
    const float bw = b[3];
    out[0] = aw * bx + ax * bw + ay * bz - az * by;
    out[1] = aw * by - ax * bz + ay * bw + az * bx;
    out[2] = aw * bz + ax * by - ay * bx + az * bw;
    out[3] = aw * bw - ax * bx - ay * by - az * bz;
}

__device__ inline void so3_exp_quat(const float phi[3], float q[4]) {
    const float theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    if (theta_sq <= 1.0e-8f) {
        const float imag_scale = 0.5f - theta_sq / 48.0f;
        q[0] = imag_scale * phi[0];
        q[1] = imag_scale * phi[1];
        q[2] = imag_scale * phi[2];
        q[3] = 1.0f - theta_sq / 8.0f;
        normalize_quat(q);
        return;
    }
    const float theta = sqrtf(theta_sq);
    const float half_theta = 0.5f * theta;
    const float imag_scale = sinf(half_theta) / theta;
    q[0] = imag_scale * phi[0];
    q[1] = imag_scale * phi[1];
    q[2] = imag_scale * phi[2];
    q[3] = cosf(half_theta);
    normalize_quat(q);
}

__device__ inline void quat_to_matrix(const float q_in[4], float R[9]) {
    float q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
    normalize_quat(q);
    const float x = q[0];
    const float y = q[1];
    const float z = q[2];
    const float w = q[3];
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xy = x * y;
    const float xz = x * z;
    const float yz = y * z;
    const float wx = w * x;
    const float wy = w * y;
    const float wz = w * z;
    R[0] = 1.0f - 2.0f * (yy + zz);
    R[1] = 2.0f * (xy - wz);
    R[2] = 2.0f * (xz + wy);
    R[3] = 2.0f * (xy + wz);
    R[4] = 1.0f - 2.0f * (xx + zz);
    R[5] = 2.0f * (yz - wx);
    R[6] = 2.0f * (xz - wy);
    R[7] = 2.0f * (yz + wx);
    R[8] = 1.0f - 2.0f * (xx + yy);
}

__device__ inline void quat_rotate(const float q_in[4], const float v[3], float out[3]) {
    float q[4] = {q_in[0], q_in[1], q_in[2], q_in[3]};
    normalize_quat(q);
    const float uv[3] = {
        q[1] * v[2] - q[2] * v[1],
        q[2] * v[0] - q[0] * v[2],
        q[0] * v[1] - q[1] * v[0],
    };
    const float uuv[3] = {
        q[1] * uv[2] - q[2] * uv[1],
        q[2] * uv[0] - q[0] * uv[2],
        q[0] * uv[1] - q[1] * uv[0],
    };
    out[0] = v[0] + 2.0f * (q[3] * uv[0] + uuv[0]);
    out[1] = v[1] + 2.0f * (q[3] * uv[1] + uuv[1]);
    out[2] = v[2] + 2.0f * (q[3] * uv[2] + uuv[2]);
}

__device__ inline void mat_vec(const float R[9], const float v[3], float out[3]) {
    out[0] = R[0] * v[0] + R[1] * v[1] + R[2] * v[2];
    out[1] = R[3] * v[0] + R[4] * v[1] + R[5] * v[2];
    out[2] = R[6] * v[0] + R[7] * v[1] + R[8] * v[2];
}

__device__ inline void mat_t_vec(const float R[9], const float v[3], float out[3]) {
    out[0] = R[0] * v[0] + R[3] * v[1] + R[6] * v[2];
    out[1] = R[1] * v[0] + R[4] * v[1] + R[7] * v[2];
    out[2] = R[2] * v[0] + R[5] * v[1] + R[8] * v[2];
}

__device__ inline void quat_log(float q[4], float out[3]) {
    normalize_quat(q);
    if (q[3] < 0.0f) {
        q[0] = -q[0];
        q[1] = -q[1];
        q[2] = -q[2];
        q[3] = -q[3];
    }
    const float vec_norm = sqrtf(q[0] * q[0] + q[1] * q[1] + q[2] * q[2]);
    if (vec_norm < EPS) {
        out[0] = 2.0f * q[0];
        out[1] = 2.0f * q[1];
        out[2] = 2.0f * q[2];
        return;
    }
    const float theta = 2.0f * atan2f(vec_norm, q[3]);
    const float scale = theta / vec_norm;
    out[0] = scale * q[0];
    out[1] = scale * q[1];
    out[2] = scale * q[2];
}

__device__ inline void so3_hat(const float v[3], float H[9]) {
    H[0] = 0.0f;
    H[1] = -v[2];
    H[2] = v[1];
    H[3] = v[2];
    H[4] = 0.0f;
    H[5] = -v[0];
    H[6] = -v[1];
    H[7] = v[0];
    H[8] = 0.0f;
}

__device__ inline void mat_mul3(const float A[9], const float B[9], float C[9]) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            C[3 * r + c] =
                A[3 * r + 0] * B[c + 0] +
                A[3 * r + 1] * B[c + 3] +
                A[3 * r + 2] * B[c + 6];
        }
    }
}

__device__ inline void right_jacobian_inverse(const float phi[3], float J[9]) {
    const float theta_sq = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    const float theta = sqrtf(fmaxf(theta_sq, EPS));
    float H[9];
    so3_hat(phi, H);
    float H2[9];
    mat_mul3(H, H, H2);
    const float sin_theta = sinf(theta);
    const float cos_theta = cosf(theta);
    const float denom = fmaxf(2.0f * theta * sin_theta, EPS);
    float coeff = 1.0f / fmaxf(theta_sq, EPS) - (1.0f + cos_theta) / denom;
    const float taylor = 1.0f / 12.0f + theta_sq / 720.0f + theta_sq * theta_sq / 30240.0f;
    if (theta_sq <= 1.0e-8f) {
        coeff = taylor;
    }
    for (int idx = 0; idx < 9; ++idx) {
        J[idx] = 0.5f * H[idx] + coeff * H2[idx];
    }
    J[0] += 1.0f;
    J[4] += 1.0f;
    J[8] += 1.0f;
}

__device__ inline void relative_translation_scale_pose(
    const float* pose_i,
    const float* pose_j,
    float pred_t[3],
    float R_j[9]) {
    // Translation+scale needs only predicted translation and R_j for d(pred_t)/d(t_j).
    // Avoid building R_i and pred_q; full SE3 below keeps matrices for rotation Jacobians.
    float q_i[4];
    float q_j[4];
    load_quat(pose_i + 3, q_i);
    load_quat(pose_j + 3, q_j);

    const float t_i[3] = {pose_i[0], pose_i[1], pose_i[2]};
    const float t_j[3] = {pose_j[0], pose_j[1], pose_j[2]};
    float q_i_inv[4];
    quat_inv(q_i, q_i_inv);
    float source_t_in_source[3];
    quat_rotate(q_i_inv, t_i, source_t_in_source);
    const float t_i_inv[3] = {-source_t_in_source[0], -source_t_in_source[1], -source_t_in_source[2]};

    quat_to_matrix(q_j, R_j);
    float rotated_t_i_inv[3];
    mat_vec(R_j, t_i_inv, rotated_t_i_inv);
    pred_t[0] = t_j[0] + rotated_t_i_inv[0];
    pred_t[1] = t_j[1] + rotated_t_i_inv[1];
    pred_t[2] = t_j[2] + rotated_t_i_inv[2];
}

__device__ inline void relative_pose(
    const float* pose_i,
    const float* pose_j,
    float pred_t[3],
    float pred_q[4],
    float R_i[9],
    float R_j[9],
    float source_t_in_source[3]) {
    float q_i[4];
    float q_j[4];
    load_quat(pose_i + 3, q_i);
    load_quat(pose_j + 3, q_j);
    quat_to_matrix(q_i, R_i);
    quat_to_matrix(q_j, R_j);

    const float t_i[3] = {pose_i[0], pose_i[1], pose_i[2]};
    const float t_j[3] = {pose_j[0], pose_j[1], pose_j[2]};
    mat_t_vec(R_i, t_i, source_t_in_source);
    const float t_i_inv[3] = {-source_t_in_source[0], -source_t_in_source[1], -source_t_in_source[2]};
    float rotated_t_i_inv[3];
    mat_vec(R_j, t_i_inv, rotated_t_i_inv);
    pred_t[0] = t_j[0] + rotated_t_i_inv[0];
    pred_t[1] = t_j[1] + rotated_t_i_inv[1];
    pred_t[2] = t_j[2] + rotated_t_i_inv[2];

    float q_i_inv[4];
    quat_inv(q_i, q_i_inv);
    quat_mul(q_j, q_i_inv, pred_q);
    normalize_quat(pred_q);
}

__device__ inline void rotation_residual_from_quats(
    const float* q_source,
    const float* q_target,
    const float* q_meas,
    float residual[3],
    float R_source[9]) {
    float source[4];
    float target[4];
    float meas[4];
    load_quat(q_source, source);
    load_quat(q_target, target);
    load_quat(q_meas, meas);
    quat_to_matrix(source, R_source);
    float source_inv[4];
    quat_inv(source, source_inv);
    float pred[4];
    quat_mul(target, source_inv, pred);
    float meas_inv[4];
    quat_inv(meas, meas_inv);
    float err[4];
    quat_mul(meas_inv, pred, err);
    quat_log(err, residual);
}

__global__ void se3_scale_apply_delta_kernel(
    const float* poses,
    const float* log_s,
    const float* step_full,
    int anchor,
    int n_nodes,
    float* candidate_poses,
    float* candidate_log_s) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_nodes) {
        return;
    }

    const float* pose = poses + idx * 7;
    const float* step = step_full + idx * 7;
    float* candidate_pose = candidate_poses + idx * 7;
    candidate_log_s[idx] = log_s[idx] + step[6];

    if (idx == anchor) {
        for (int k = 0; k < 7; ++k) {
            candidate_pose[k] = pose[k];
        }
        return;
    }

    float q_current[4];
    load_quat(pose + 3, q_current);
    float rotated_dt[3];
    const float dt[3] = {step[0], step[1], step[2]};
    quat_rotate(q_current, dt, rotated_dt);
    candidate_pose[0] = pose[0] + rotated_dt[0];
    candidate_pose[1] = pose[1] + rotated_dt[1];
    candidate_pose[2] = pose[2] + rotated_dt[2];

    const float dphi[3] = {step[3], step[4], step[5]};
    float q_delta[4];
    so3_exp_quat(dphi, q_delta);
    float q_new[4];
    quat_mul(q_current, q_delta, q_new);
    normalize_quat(q_new);
    candidate_pose[3] = q_new[0];
    candidate_pose[4] = q_new[1];
    candidate_pose[5] = q_new[2];
    candidate_pose[6] = q_new[3];
}

__global__ void se3_scale_candidate_summary_kernel(
    const float* candidate_poses,
    const float* candidate_log_s,
    const float* step_full,
    const float* rel_poses,
    const float* prior_log_s,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    float scale_prior_diag,
    int n_nodes,
    int n_edges,
    float* summary) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_edges) {
        const int i = static_cast<int>(ii[idx]);
        const int j = static_cast<int>(jj[idx]);
        const float* pose_i = candidate_poses + i * 7;
        const float* pose_j = candidate_poses + j * 7;
        const float* rel_pose = rel_poses + idx * 7;

        float pred_metric[3];
        float pred_q[4];
        float R_i[9];
        float R_j[9];
        float source_t_in_source[3];
        relative_pose(pose_i, pose_j, pred_metric, pred_q, R_i, R_j, source_t_in_source);
        const float inv_source_scale = 1.0f / clamp_scale(expf(candidate_log_s[i]));

        float rel_q[4];
        float rel_q_inv[4];
        load_quat(rel_pose + 3, rel_q);
        quat_inv(rel_q, rel_q_inv);
        float rot_err_q[4];
        quat_mul(rel_q_inv, pred_q, rot_err_q);
        float rot_residual[3];
        quat_log(rot_err_q, rot_residual);

        const float residual[6] = {
            pred_metric[0] * inv_source_scale - rel_pose[0],
            pred_metric[1] * inv_source_scale - rel_pose[1],
            pred_metric[2] * inv_source_scale - rel_pose[2],
            rot_residual[0],
            rot_residual[1],
            rot_residual[2],
        };

        const float robust_e = robust[idx];
        float edge_cost = 0.0f;
        for (int r = 0; r < 6; ++r) {
            const float weighted = residual[r] * sqrt_info[idx * 6 + r] * robust_e;
            edge_cost += weighted * weighted;
        }
        atomicAdd(summary, 0.5f * edge_cost);
    }

    if (idx < n_nodes) {
        const float prior = candidate_log_s[idx] - prior_log_s[idx];
        atomicAdd(summary, 0.5f * prior * prior * scale_prior_diag);

        float step_sq = 0.0f;
        const float* step = step_full + idx * 7;
        for (int k = 0; k < 7; ++k) {
            step_sq += step[k] * step[k];
        }
        atomicAdd(summary + 1, step_sq);
    }
}

__global__ void se3_scale_stats_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const float* prior_log_s,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    float scale_prior_diag,
    int n_nodes,
    int n_edges,
    float* summary) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_edges) {
        const int i = static_cast<int>(ii[idx]);
        const int j = static_cast<int>(jj[idx]);
        const float* pose_i = poses + i * 7;
        const float* pose_j = poses + j * 7;
        const float* rel_pose = rel_poses + idx * 7;

        float pred_metric[3];
        float pred_q[4];
        float R_i[9];
        float R_j[9];
        float source_t_in_source[3];
        relative_pose(pose_i, pose_j, pred_metric, pred_q, R_i, R_j, source_t_in_source);
        const float inv_source_scale = 1.0f / clamp_scale(expf(log_s[i]));

        float rel_q[4];
        float rel_q_inv[4];
        load_quat(rel_pose + 3, rel_q);
        quat_inv(rel_q, rel_q_inv);
        float rot_err_q[4];
        quat_mul(rel_q_inv, pred_q, rot_err_q);
        float rot_residual[3];
        quat_log(rot_err_q, rot_residual);

        const float residual[6] = {
            pred_metric[0] * inv_source_scale - rel_pose[0],
            pred_metric[1] * inv_source_scale - rel_pose[1],
            pred_metric[2] * inv_source_scale - rel_pose[2],
            rot_residual[0],
            rot_residual[1],
            rot_residual[2],
        };

        float weighted_sq = 0.0f;
        bool finite = true;
        for (int r = 0; r < 6; ++r) {
            const float weighted = residual[r] * sqrt_info[idx * 6 + r];
            weighted_sq += weighted * weighted;
            finite = finite && isfinite(weighted);
        }
        const float edge_norm = sqrtf(weighted_sq);
        finite = finite && isfinite(edge_norm);
        if (finite) {
            atomicAdd(summary, 0.5f * weighted_sq);
            atomicAdd(summary + 1, edge_norm);
            atomic_max_float(summary + 2, edge_norm);
        } else {
            atomicAdd(summary + 5, 1.0f);
        }
    }

    if (idx < n_nodes) {
        const float scale_sqrt_info = sqrtf(scale_prior_diag);
        const float prior = (log_s[idx] - prior_log_s[idx]) * scale_sqrt_info;
        if (isfinite(prior)) {
            const float prior_abs = fabsf(prior);
            atomicAdd(summary, 0.5f * prior * prior);
            atomicAdd(summary + 3, prior_abs);
            atomic_max_float(summary + 4, prior_abs);
        } else {
            atomicAdd(summary + 5, 1.0f);
        }
    }
}

__global__ void rotation_blocks_kernel(
    const float* rotations,
    const float* meas_rotations,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_edges) {
        return;
    }
    const int i = static_cast<int>(ii[e]);
    const int j = static_cast<int>(jj[e]);
    float residual[3];
    float R_i[9];
    rotation_residual_from_quats(rotations + i * 4, rotations + j * 4, meas_rotations + e * 4, residual, R_i);
    float Jinv[9];
    right_jacobian_inverse(residual, Jinv);
    float block[9];
    mat_mul3(Jinv, R_i, block);
    const float robust_e = robust[e];
    for (int r = 0; r < 3; ++r) {
        const float weight = sqrt_info[e * 3 + r] * robust_e;
        edge_residual[e * 3 + r] = residual[r] * weight;
        for (int c = 0; c < 3; ++c) {
            const float value = block[3 * r + c] * weight;
            source_block[e * 9 + 3 * r + c] = -value;
            target_block[e * 9 + 3 * r + c] = value;
        }
    }
}

__global__ void translation_scale_blocks_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const float* prior_log_s,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    float scale_prior_diag,
    int n_nodes,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual,
    float* prior_gradient) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_edges) {
        const int i = static_cast<int>(ii[idx]);
        const int j = static_cast<int>(jj[idx]);
        const float* pose_i = poses + i * 7;
        const float* pose_j = poses + j * 7;
        float pred_metric[3];
        float R_j[9];
        relative_translation_scale_pose(pose_i, pose_j, pred_metric, R_j);
        const float inv_source_scale = 1.0f / clamp_scale(expf(log_s[i]));
        const float robust_e = robust[idx];
        for (int r = 0; r < 3; ++r) {
            const float pred_t = pred_metric[r] * inv_source_scale;
            const float residual = pred_t - rel_poses[idx * 7 + r];
            const float weight = sqrt_info[idx * 3 + r] * robust_e;
            edge_residual[idx * 3 + r] = residual * weight;
            for (int c = 0; c < 4; ++c) {
                source_block[idx * 12 + r * 4 + c] = 0.0f;
                target_block[idx * 12 + r * 4 + c] = 0.0f;
            }
            for (int c = 0; c < 3; ++c) {
                const float value = R_j[3 * r + c] * inv_source_scale * weight;
                source_block[idx * 12 + r * 4 + c] = -value;
                target_block[idx * 12 + r * 4 + c] = value;
            }
            source_block[idx * 12 + r * 4 + 3] = -pred_t * weight;
        }
    }
    if (idx < n_nodes) {
        prior_gradient[idx] = (log_s[idx] - prior_log_s[idx]) * scale_prior_diag;
    }
}

__global__ void se3_scale_blocks_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const float* prior_log_s,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    float scale_prior_diag,
    int n_nodes,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual,
    float* prior_gradient) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_edges) {
        const int i = static_cast<int>(ii[idx]);
        const int j = static_cast<int>(jj[idx]);
        const float* pose_i = poses + i * 7;
        const float* pose_j = poses + j * 7;
        const float* rel_pose = rel_poses + idx * 7;
        float pred_metric[3];
        float pred_q[4];
        float R_i[9];
        float R_j[9];
        float source_t_in_source[3];
        relative_pose(pose_i, pose_j, pred_metric, pred_q, R_i, R_j, source_t_in_source);
        const float inv_source_scale = 1.0f / clamp_scale(expf(log_s[i]));
        float pred_t[3] = {
            pred_metric[0] * inv_source_scale,
            pred_metric[1] * inv_source_scale,
            pred_metric[2] * inv_source_scale,
        };

        float rel_q_inv[4];
        float rel_q[4];
        load_quat(rel_pose + 3, rel_q);
        quat_inv(rel_q, rel_q_inv);
        float rot_err_q[4];
        quat_mul(rel_q_inv, pred_q, rot_err_q);
        float rot_residual[3];
        quat_log(rot_err_q, rot_residual);

        float residual[6] = {
            pred_t[0] - rel_pose[0],
            pred_t[1] - rel_pose[1],
            pred_t[2] - rel_pose[2],
            rot_residual[0],
            rot_residual[1],
            rot_residual[2],
        };

        for (int k = 0; k < 42; ++k) {
            source_block[idx * 42 + k] = 0.0f;
            target_block[idx * 42 + k] = 0.0f;
        }

        float source_hat[9];
        so3_hat(source_t_in_source, source_hat);
        float trans_rot_block[9];
        mat_mul3(R_j, source_hat, trans_rot_block);
        float Jinv[9];
        right_jacobian_inverse(rot_residual, Jinv);
        float rotation_block[9];
        mat_mul3(Jinv, R_i, rotation_block);

        const float robust_e = robust[idx];
        for (int r = 0; r < 6; ++r) {
            const float weight = sqrt_info[idx * 6 + r] * robust_e;
            edge_residual[idx * 6 + r] = residual[r] * weight;
            if (r < 3) {
                for (int c = 0; c < 3; ++c) {
                    const float translation_value = R_j[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 42 + r * 7 + c] = -translation_value;
                    target_block[idx * 42 + r * 7 + c] = translation_value;

                    const float rotation_value = trans_rot_block[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 42 + r * 7 + 3 + c] = -rotation_value;
                    target_block[idx * 42 + r * 7 + 3 + c] = rotation_value;
                }
                source_block[idx * 42 + r * 7 + 6] = -pred_t[r] * weight;
            } else {
                const int rr = r - 3;
                for (int c = 0; c < 3; ++c) {
                    const float value = rotation_block[3 * rr + c] * weight;
                    source_block[idx * 42 + r * 7 + 3 + c] = -value;
                    target_block[idx * 42 + r * 7 + 3 + c] = value;
                }
            }
        }
    }
    if (idx < n_nodes) {
        prior_gradient[idx] = (log_s[idx] - prior_log_s[idx]) * scale_prior_diag;
    }
}

__global__ void se3_scale_weighted_blocks_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const float* prior_log_s,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    float huber_delta,
    float scale_prior_diag,
    int n_nodes,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual,
    float* prior_gradient,
    float* robust,
    float* summary) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_edges) {
        const int i = static_cast<int>(ii[idx]);
        const int j = static_cast<int>(jj[idx]);
        const float* pose_i = poses + i * 7;
        const float* pose_j = poses + j * 7;
        const float* rel_pose = rel_poses + idx * 7;
        float pred_metric[3];
        float pred_q[4];
        float R_i[9];
        float R_j[9];
        float source_t_in_source[3];
        relative_pose(pose_i, pose_j, pred_metric, pred_q, R_i, R_j, source_t_in_source);
        const float inv_source_scale = 1.0f / clamp_scale(expf(log_s[i]));
        float pred_t[3] = {
            pred_metric[0] * inv_source_scale,
            pred_metric[1] * inv_source_scale,
            pred_metric[2] * inv_source_scale,
        };

        float rel_q_inv[4];
        float rel_q[4];
        load_quat(rel_pose + 3, rel_q);
        quat_inv(rel_q, rel_q_inv);
        float rot_err_q[4];
        quat_mul(rel_q_inv, pred_q, rot_err_q);
        float rot_residual[3];
        quat_log(rot_err_q, rot_residual);

        float residual[6] = {
            pred_t[0] - rel_pose[0],
            pred_t[1] - rel_pose[1],
            pred_t[2] - rel_pose[2],
            rot_residual[0],
            rot_residual[1],
            rot_residual[2],
        };

        float unrobust_sq = 0.0f;
        for (int r = 0; r < 6; ++r) {
            const float weighted = residual[r] * sqrt_info[idx * 6 + r];
            unrobust_sq += weighted * weighted;
        }
        const float edge_norm = sqrtf(fmaxf(unrobust_sq, EPS));
        float robust_e = 1.0f;
        if (edge_norm > huber_delta) {
            robust_e = sqrtf(huber_delta / edge_norm);
        }
        robust[idx] = robust_e;
        atomicAdd(summary + 1, 0.5f * unrobust_sq);
        atomicAdd(summary, 0.5f * unrobust_sq * robust_e * robust_e);

        for (int k = 0; k < 42; ++k) {
            source_block[idx * 42 + k] = 0.0f;
            target_block[idx * 42 + k] = 0.0f;
        }

        float source_hat[9];
        so3_hat(source_t_in_source, source_hat);
        float trans_rot_block[9];
        mat_mul3(R_j, source_hat, trans_rot_block);
        float Jinv[9];
        right_jacobian_inverse(rot_residual, Jinv);
        float rotation_block[9];
        mat_mul3(Jinv, R_i, rotation_block);

        for (int r = 0; r < 6; ++r) {
            const float weight = sqrt_info[idx * 6 + r] * robust_e;
            edge_residual[idx * 6 + r] = residual[r] * weight;
            if (r < 3) {
                for (int c = 0; c < 3; ++c) {
                    const float translation_value = R_j[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 42 + r * 7 + c] = -translation_value;
                    target_block[idx * 42 + r * 7 + c] = translation_value;

                    const float rotation_value = trans_rot_block[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 42 + r * 7 + 3 + c] = -rotation_value;
                    target_block[idx * 42 + r * 7 + 3 + c] = rotation_value;
                }
                source_block[idx * 42 + r * 7 + 6] = -pred_t[r] * weight;
            } else {
                const int rr = r - 3;
                for (int c = 0; c < 3; ++c) {
                    const float value = rotation_block[3 * rr + c] * weight;
                    source_block[idx * 42 + r * 7 + 3 + c] = -value;
                    target_block[idx * 42 + r * 7 + 3 + c] = value;
                }
            }
        }
    }
    if (idx < n_nodes) {
        const float prior = log_s[idx] - prior_log_s[idx];
        prior_gradient[idx] = prior * scale_prior_diag;
        const float prior_cost = 0.5f * prior * prior * scale_prior_diag;
        atomicAdd(summary, prior_cost);
        atomicAdd(summary + 1, prior_cost);
    }
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> rotation_blocks_cuda(
    torch::Tensor rotations,
    torch::Tensor meas_rotations,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust) {
    CHECK_CUDA(rotations);
    CHECK_CUDA(meas_rotations);
    CHECK_FLOAT(rotations);
    CHECK_FLOAT(meas_rotations);
    CHECK_CONTIGUOUS(rotations);
    CHECK_CONTIGUOUS(meas_rotations);
    TORCH_CHECK(rotations.dim() == 2 && rotations.size(1) == 4, "rotations must have shape (N, 4)");
    TORCH_CHECK(meas_rotations.dim() == 2 && meas_rotations.size(1) == 4, "meas_rotations must have shape (E, 4)");
    const int64_t n_edges64 = meas_rotations.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 3);

    const c10::cuda::CUDAGuard device_guard(rotations.device());
    auto source_block = torch::empty({n_edges64, 3, 3}, rotations.options());
    auto target_block = torch::empty({n_edges64, 3, 3}, rotations.options());
    auto edge_residual = torch::empty({n_edges64, 3}, rotations.options());
    const int n_edges = static_cast<int>(n_edges64);
    if (n_edges > 0) {
        const int blocks = (n_edges + THREADS - 1) / THREADS;
        rotation_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            rotations.data_ptr<float>(),
            meas_rotations.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            n_edges,
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {source_block, target_block, edge_residual};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> translation_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust,
    double scale_prior_diag) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(prior_log_s);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(prior_log_s);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(prior_log_s);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(prior_log_s.dim() == 1 && prior_log_s.numel() == poses.size(0), "prior_log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 3);

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto source_block = torch::empty({n_edges64, 3, 4}, poses.options());
    auto target_block = torch::empty({n_edges64, 3, 4}, poses.options());
    auto edge_residual = torch::empty({n_edges64, 3}, poses.options());
    auto prior_gradient = torch::empty({n_nodes64}, poses.options());
    const int total = static_cast<int>(std::max(n_nodes64, n_edges64));
    if (total > 0) {
        const int blocks = (total + THREADS - 1) / THREADS;
        translation_scale_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            prior_log_s.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            static_cast<float>(scale_prior_diag),
            static_cast<int>(n_nodes64),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>(),
            prior_gradient.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {source_block, target_block, edge_residual, prior_gradient};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> se3_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust,
    double scale_prior_diag) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(prior_log_s);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(prior_log_s);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(prior_log_s);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(prior_log_s.dim() == 1 && prior_log_s.numel() == poses.size(0), "prior_log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 6);

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto source_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto target_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto edge_residual = torch::empty({n_edges64, 6}, poses.options());
    auto prior_gradient = torch::empty({n_nodes64}, poses.options());
    const int total = static_cast<int>(std::max(n_nodes64, n_edges64));
    if (total > 0) {
        const int blocks = (total + THREADS - 1) / THREADS;
        se3_scale_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            prior_log_s.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            static_cast<float>(scale_prior_diag),
            static_cast<int>(n_nodes64),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>(),
            prior_gradient.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {source_block, target_block, edge_residual, prior_gradient};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double, double>
se3_scale_weighted_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    double huber_delta,
    double scale_prior_diag) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(prior_log_s);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(prior_log_s);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(prior_log_s);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(prior_log_s.dim() == 1 && prior_log_s.numel() == poses.size(0), "prior_log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    CHECK_CUDA(sqrt_info);
    CHECK_FLOAT(sqrt_info);
    CHECK_CONTIGUOUS(sqrt_info);
    TORCH_CHECK(sqrt_info.dim() == 2 && sqrt_info.size(0) == n_edges64 && sqrt_info.size(1) == 6,
                "sqrt_info must have shape (E, 6)");

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto source_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto target_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto edge_residual = torch::empty({n_edges64, 6}, poses.options());
    auto prior_gradient = torch::empty({n_nodes64}, poses.options());
    auto robust = torch::empty({n_edges64, 1}, poses.options());
    auto summary = torch::zeros({2}, poses.options());
    const int total = static_cast<int>(std::max(n_nodes64, n_edges64));
    if (total > 0) {
        const int blocks = (total + THREADS - 1) / THREADS;
        se3_scale_weighted_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            prior_log_s.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            static_cast<float>(huber_delta),
            static_cast<float>(scale_prior_diag),
            static_cast<int>(n_nodes64),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>(),
            prior_gradient.data_ptr<float>(),
            robust.data_ptr<float>(),
            summary.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto summary_cpu = summary.to(torch::kCPU);
    const float* summary_ptr = summary_cpu.data_ptr<float>();
    return {
        source_block,
        target_block,
        edge_residual,
        prior_gradient,
        robust,
        static_cast<double>(summary_ptr[0]),
        static_cast<double>(summary_ptr[1]),
    };
}

std::tuple<torch::Tensor, torch::Tensor, double, double> se3_scale_candidate_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor step_full,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust,
    int64_t anchor,
    double scale_prior_diag) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(step_full);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(prior_log_s);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(step_full);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(prior_log_s);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(step_full);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(prior_log_s);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(step_full.dim() == 2 && step_full.sizes() == poses.sizes(), "step_full must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(prior_log_s.dim() == 1 && prior_log_s.numel() == poses.size(0), "prior_log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = rel_poses.size(0);
    TORCH_CHECK(anchor >= 0 && anchor < n_nodes64, "anchor out of range");
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 6);

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto candidate_poses = torch::empty_like(poses);
    auto candidate_log_s = torch::empty_like(log_s);
    auto summary = torch::zeros({2}, poses.options());
    const int n_nodes = static_cast<int>(n_nodes64);
    const int n_edges = static_cast<int>(n_edges64);
    if (n_nodes > 0) {
        const int blocks = (n_nodes + THREADS - 1) / THREADS;
        se3_scale_apply_delta_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            step_full.data_ptr<float>(),
            static_cast<int>(anchor),
            n_nodes,
            candidate_poses.data_ptr<float>(),
            candidate_log_s.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    const int total = static_cast<int>(std::max(n_nodes64, n_edges64));
    if (total > 0) {
        const int blocks = (total + THREADS - 1) / THREADS;
        se3_scale_candidate_summary_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            candidate_poses.data_ptr<float>(),
            candidate_log_s.data_ptr<float>(),
            step_full.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            prior_log_s.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            static_cast<float>(scale_prior_diag),
            n_nodes,
            n_edges,
            summary.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto summary_cpu = summary.to(torch::kCPU);
    const float* summary_ptr = summary_cpu.data_ptr<float>();
    return {
        candidate_poses,
        candidate_log_s,
        static_cast<double>(summary_ptr[0]),
        static_cast<double>(std::sqrt(static_cast<double>(summary_ptr[1]))),
    };
}

std::tuple<double, double, double, double, double, bool> se3_scale_stats_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor prior_log_s,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    double scale_prior_diag) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(prior_log_s);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(prior_log_s);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(prior_log_s);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(prior_log_s.dim() == 1 && prior_log_s.numel() == poses.size(0), "prior_log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    CHECK_CUDA(sqrt_info);
    CHECK_FLOAT(sqrt_info);
    CHECK_CONTIGUOUS(sqrt_info);
    TORCH_CHECK(sqrt_info.dim() == 2 && sqrt_info.size(0) == n_edges64 && sqrt_info.size(1) == 6,
                "sqrt_info must have shape (E, 6)");

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto summary = torch::zeros({6}, poses.options());
    const int total = static_cast<int>(std::max(n_nodes64, n_edges64));
    if (total > 0) {
        const int blocks = (total + THREADS - 1) / THREADS;
        se3_scale_stats_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            prior_log_s.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            static_cast<float>(scale_prior_diag),
            static_cast<int>(n_nodes64),
            static_cast<int>(n_edges64),
            summary.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }

    auto summary_cpu = summary.to(torch::kCPU);
    const float* values = summary_cpu.data_ptr<float>();
    const double edge_mean = n_edges64 > 0 ? static_cast<double>(values[1]) / static_cast<double>(n_edges64) : 0.0;
    const double prior_mean = n_nodes64 > 0 ? static_cast<double>(values[3]) / static_cast<double>(n_nodes64) : 0.0;
    return {
        static_cast<double>(values[0]),
        edge_mean,
        static_cast<double>(values[2]),
        prior_mean,
        static_cast<double>(values[4]),
        values[5] == 0.0f,
    };
}
