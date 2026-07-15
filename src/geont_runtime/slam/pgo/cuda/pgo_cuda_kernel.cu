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
#define CHECK_INT(x) TORCH_CHECK(x.scalar_type() == at::kInt, #x " must be int32")
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
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual) {
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
}

__global__ void se3_scale_blocks_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* robust,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual) {
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
}

__global__ void se3_scale_weighted_blocks_kernel(
    const float* poses,
    const float* log_s,
    const float* rel_poses,
    const float* rel_log_scales,
    const int64_t* ii,
    const int64_t* jj,
    const float* sqrt_info,
    const float* scale_sqrt_info,
    float huber_delta,
    int n_edges,
    float* source_block,
    float* target_block,
    float* edge_residual,
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

        for (int k = 0; k < 49; ++k) {
            source_block[idx * 49 + k] = 0.0f;
            target_block[idx * 49 + k] = 0.0f;
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
            edge_residual[idx * 7 + r] = residual[r] * weight;
            if (r < 3) {
                for (int c = 0; c < 3; ++c) {
                    const float translation_value = R_j[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 49 + r * 7 + c] = -translation_value;
                    target_block[idx * 49 + r * 7 + c] = translation_value;

                    const float rotation_value = trans_rot_block[3 * r + c] * inv_source_scale * weight;
                    source_block[idx * 49 + r * 7 + 3 + c] = -rotation_value;
                    target_block[idx * 49 + r * 7 + 3 + c] = rotation_value;
                }
                source_block[idx * 49 + r * 7 + 6] = -pred_t[r] * weight;
            } else {
                const int rr = r - 3;
                for (int c = 0; c < 3; ++c) {
                    const float value = rotation_block[3 * rr + c] * weight;
                    source_block[idx * 49 + r * 7 + 3 + c] = -value;
                    target_block[idx * 49 + r * 7 + 3 + c] = value;
                }
            }
        }
        const float scale_weight = scale_sqrt_info[idx];
        source_block[idx * 49 + 48] = -scale_weight;
        target_block[idx * 49 + 48] = scale_weight;
        const float scale_residual =
            (log_s[j] - log_s[i] - rel_log_scales[idx]) * scale_weight;
        edge_residual[idx * 7 + 6] = scale_residual;
    }
}

__device__ inline double se3_scale_column_dot(
    const float* block_a,
    const float* block_b,
    int col_a,
    int col_b,
    int residual_mask) {
    double value = 0.0;
    for (int row = 0; row < 7; ++row) {
        if ((residual_mask & (1 << row)) != 0) {
            const double product = __dmul_rn(
                static_cast<double>(block_a[row * 7 + col_a]),
                static_cast<double>(block_b[row * 7 + col_b]));
            value = __dadd_rn(value, product);
        }
    }
    return value;
}

__global__ void se3_scale_hessian_values_kernel(
    const float* source_block,
    const float* target_block,
    const int* group_offsets,
    const int64_t* metadata,
    int n_values,
    double* values) {
    const int value_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (value_index >= n_values) {
        return;
    }
    double value = 0.0;
    for (int slot = group_offsets[value_index]; slot < group_offsets[value_index + 1]; ++slot) {
        const uint64_t item = static_cast<uint64_t>(metadata[slot]);
        const int edge = static_cast<int>(item & 0xffffffffu);
        const int block_a_kind = static_cast<int>((item >> 32) & 1u);
        const int block_b_kind = static_cast<int>((item >> 33) & 1u);
        const int col_a = static_cast<int>((item >> 34) & 7u);
        const int col_b = static_cast<int>((item >> 37) & 7u);
        const int residual_mask = static_cast<int>((item >> 40) & 0x7fu);
        const float* block_a = (block_a_kind == 0 ? source_block : target_block) + edge * 49;
        const float* block_b = (block_b_kind == 0 ? source_block : target_block) + edge * 49;
        value = __dadd_rn(
            value,
            se3_scale_column_dot(block_a, block_b, col_a, col_b, residual_mask));
    }
    values[value_index] = value;
}

__global__ void se3_scale_gradient_kernel(
    const float* source_block,
    const float* target_block,
    const float* edge_residual,
    const int* group_offsets,
    const int64_t* metadata,
    int n_variables,
    double* gradient) {
    const int variable = blockIdx.x * blockDim.x + threadIdx.x;
    if (variable >= n_variables) {
        return;
    }
    double value = 0.0;
    for (int slot = group_offsets[variable]; slot < group_offsets[variable + 1]; ++slot) {
        const uint64_t item = static_cast<uint64_t>(metadata[slot]);
        const int edge = static_cast<int>(item & 0xffffffffu);
        const int block_kind = static_cast<int>((item >> 32) & 1u);
        const float* block = (block_kind == 0 ? source_block : target_block) + edge * 49;
        const float* residual = edge_residual + edge * 7;
        const int col = static_cast<int>((item >> 33) & 7u);
        const int residual_mask = static_cast<int>((item >> 36) & 0x7fu);
        double contribution = 0.0;
        for (int row = 0; row < 7; ++row) {
            if ((residual_mask & (1 << row)) != 0) {
                const double product = __dmul_rn(
                    static_cast<double>(block[row * 7 + col]),
                    static_cast<double>(residual[row]));
                contribution = __dadd_rn(contribution, product);
            }
        }
        value = __dadd_rn(value, contribution);
    }
    gradient[variable] = value;
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> translation_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 3);

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto source_block = torch::empty({n_edges64, 3, 4}, poses.options());
    auto target_block = torch::empty({n_edges64, 3, 4}, poses.options());
    auto edge_residual = torch::empty({n_edges64, 3}, poses.options());
    if (n_edges64 > 0) {
        const int blocks = (n_edges64 + THREADS - 1) / THREADS;
        translation_scale_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {source_block, target_block, edge_residual};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> se3_scale_blocks_cuda(
    torch::Tensor poses,
    torch::Tensor log_s,
    torch::Tensor rel_poses,
    torch::Tensor ii,
    torch::Tensor jj,
    torch::Tensor sqrt_info,
    torch::Tensor robust) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_edges64 = rel_poses.size(0);
    check_edge_index_tensors(ii, jj, n_edges64);
    check_weights(sqrt_info, robust, n_edges64, 6);

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto source_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto target_block = torch::empty({n_edges64, 6, 7}, poses.options());
    auto edge_residual = torch::empty({n_edges64, 6}, poses.options());
    if (n_edges64 > 0) {
        const int blocks = (n_edges64 + THREADS - 1) / THREADS;
        se3_scale_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            robust.data_ptr<float>(),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {source_block, target_block, edge_residual};
}

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
    torch::Tensor summary) {
    CHECK_CUDA(poses);
    CHECK_CUDA(log_s);
    CHECK_CUDA(rel_poses);
    CHECK_CUDA(rel_log_scales);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(log_s);
    CHECK_FLOAT(rel_poses);
    CHECK_FLOAT(rel_log_scales);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(log_s);
    CHECK_CONTIGUOUS(rel_poses);
    CHECK_CONTIGUOUS(rel_log_scales);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(log_s.dim() == 1 && log_s.numel() == poses.size(0), "log_s must have shape (N,)");
    TORCH_CHECK(rel_poses.dim() == 2 && rel_poses.size(1) == 7, "rel_poses must have shape (E, 7)");
    const int64_t n_edges64 = rel_poses.size(0);
    TORCH_CHECK(rel_log_scales.dim() == 1 && rel_log_scales.numel() == n_edges64,
                "rel_log_scales must have shape (E,)");
    check_edge_index_tensors(ii, jj, n_edges64);
    CHECK_CUDA(sqrt_info);
    CHECK_CUDA(scale_sqrt_info);
    CHECK_FLOAT(sqrt_info);
    CHECK_FLOAT(scale_sqrt_info);
    CHECK_CONTIGUOUS(sqrt_info);
    CHECK_CONTIGUOUS(scale_sqrt_info);
    TORCH_CHECK(sqrt_info.dim() == 2 && sqrt_info.size(0) == n_edges64 && sqrt_info.size(1) == 6,
                "sqrt_info must have shape (E, 6)");
    TORCH_CHECK(scale_sqrt_info.dim() == 1 && scale_sqrt_info.numel() == n_edges64,
                "scale_sqrt_info must have shape (E,)");
    CHECK_CUDA(source_block);
    CHECK_CUDA(target_block);
    CHECK_CUDA(edge_residual);
    CHECK_CUDA(robust);
    CHECK_CUDA(summary);
    CHECK_FLOAT(source_block);
    CHECK_FLOAT(target_block);
    CHECK_FLOAT(edge_residual);
    CHECK_FLOAT(robust);
    CHECK_FLOAT(summary);
    CHECK_CONTIGUOUS(source_block);
    CHECK_CONTIGUOUS(target_block);
    CHECK_CONTIGUOUS(edge_residual);
    CHECK_CONTIGUOUS(robust);
    CHECK_CONTIGUOUS(summary);
    TORCH_CHECK(source_block.sizes() == torch::IntArrayRef({n_edges64, 7, 7}),
                "source_block must have shape (E, 7, 7)");
    TORCH_CHECK(target_block.sizes() == source_block.sizes(), "target_block shape mismatch");
    TORCH_CHECK(edge_residual.sizes() == torch::IntArrayRef({n_edges64, 7}),
                "edge_residual must have shape (E, 7)");
    TORCH_CHECK(robust.sizes() == torch::IntArrayRef({n_edges64, 1}),
                "robust must have shape (E, 1)");
    TORCH_CHECK(summary.dim() == 1 && summary.numel() == 2, "summary must have shape (2,)");

    const c10::cuda::CUDAGuard device_guard(poses.device());
    C10_CUDA_CHECK(cudaMemsetAsync(
        summary.data_ptr<float>(),
        0,
        2 * sizeof(float),
        at::cuda::getCurrentCUDAStream()));
    if (n_edges64 > 0) {
        const int blocks = (n_edges64 + THREADS - 1) / THREADS;
        se3_scale_weighted_blocks_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            log_s.data_ptr<float>(),
            rel_poses.data_ptr<float>(),
            rel_log_scales.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            sqrt_info.data_ptr<float>(),
            scale_sqrt_info.data_ptr<float>(),
            static_cast<float>(huber_delta),
            static_cast<int>(n_edges64),
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>(),
            robust.data_ptr<float>(),
            summary.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void se3_scale_normal_equations_cuda(
    torch::Tensor source_block,
    torch::Tensor target_block,
    torch::Tensor edge_residual,
    torch::Tensor hessian_group_offsets,
    torch::Tensor hessian_metadata,
    torch::Tensor gradient_group_offsets,
    torch::Tensor gradient_metadata,
    torch::Tensor normal) {
    CHECK_CUDA(source_block);
    CHECK_CUDA(target_block);
    CHECK_CUDA(edge_residual);
    CHECK_CUDA(hessian_group_offsets);
    CHECK_CUDA(hessian_metadata);
    CHECK_CUDA(gradient_group_offsets);
    CHECK_CUDA(gradient_metadata);
    CHECK_CUDA(normal);
    CHECK_FLOAT(source_block);
    CHECK_FLOAT(target_block);
    CHECK_FLOAT(edge_residual);
    CHECK_INT(hessian_group_offsets);
    CHECK_LONG(hessian_metadata);
    CHECK_INT(gradient_group_offsets);
    CHECK_LONG(gradient_metadata);
    TORCH_CHECK(normal.scalar_type() == at::kDouble, "normal must be float64");
    CHECK_CONTIGUOUS(source_block);
    CHECK_CONTIGUOUS(target_block);
    CHECK_CONTIGUOUS(edge_residual);
    CHECK_CONTIGUOUS(hessian_group_offsets);
    CHECK_CONTIGUOUS(hessian_metadata);
    CHECK_CONTIGUOUS(gradient_group_offsets);
    CHECK_CONTIGUOUS(gradient_metadata);
    CHECK_CONTIGUOUS(normal);
    TORCH_CHECK(source_block.dim() == 3 && source_block.size(1) == 7 && source_block.size(2) == 7,
                "source_block must have shape (E, 7, 7)");
    TORCH_CHECK(target_block.sizes() == source_block.sizes(), "target_block shape mismatch");
    TORCH_CHECK(edge_residual.dim() == 2 && edge_residual.size(0) == source_block.size(0) &&
                edge_residual.size(1) == 7, "edge_residual must have shape (E, 7)");
    TORCH_CHECK(hessian_group_offsets.dim() == 1 && hessian_group_offsets.numel() >= 1,
                "hessian_group_offsets must be 1-D");
    TORCH_CHECK(hessian_metadata.dim() == 1, "hessian_metadata must be 1-D");
    TORCH_CHECK(gradient_group_offsets.dim() == 1 && gradient_group_offsets.numel() >= 1,
                "gradient_group_offsets must be 1-D");
    TORCH_CHECK(gradient_metadata.dim() == 1, "gradient_metadata must be 1-D");

    const c10::cuda::CUDAGuard device_guard(source_block.device());
    const int n_values = static_cast<int>(hessian_group_offsets.numel() - 1);
    const int n_variables = static_cast<int>(gradient_group_offsets.numel() - 1);
    TORCH_CHECK(normal.dim() == 1 && normal.numel() == n_values + n_variables,
                "normal buffer size mismatch");
    double* values = normal.data_ptr<double>();
    double* gradient = values + n_values;
    if (n_values > 0) {
        const int blocks = (n_values + THREADS - 1) / THREADS;
        se3_scale_hessian_values_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            hessian_group_offsets.data_ptr<int>(),
            hessian_metadata.data_ptr<int64_t>(),
            n_values,
            values);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (n_variables > 0) {
        const int blocks = (n_variables + THREADS - 1) / THREADS;
        se3_scale_gradient_kernel<<<blocks, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            source_block.data_ptr<float>(),
            target_block.data_ptr<float>(),
            edge_residual.data_ptr<float>(),
            gradient_group_offsets.data_ptr<int>(),
            gradient_metadata.data_ptr<int64_t>(),
            n_variables,
            gradient);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}
