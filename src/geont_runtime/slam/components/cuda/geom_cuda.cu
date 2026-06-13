#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int THREADS = 256;
constexpr float EPS = 1.0e-8f;
constexpr float MIN_DEPTH = 0.2f;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == at::kFloat, #x " must be float32")
#define CHECK_BOOL(x) TORCH_CHECK(x.scalar_type() == at::kBool, #x " must be bool")
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

__device__ inline void relative_pose(
    const float* pose_i,
    const float* pose_j,
    float pred_t[3],
    float pred_q[4]) {
    float q_i[4];
    float q_j[4];
    float R_i[9];
    float R_j[9];
    load_quat(pose_i + 3, q_i);
    load_quat(pose_j + 3, q_j);
    quat_to_matrix(q_i, R_i);
    quat_to_matrix(q_j, R_j);

    const float t_i[3] = {pose_i[0], pose_i[1], pose_i[2]};
    const float t_j[3] = {pose_j[0], pose_j[1], pose_j[2]};
    float source_t_in_source[3];
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

__global__ void projection_distance_kernel(
    const float* poses,
    const float* depths,
    const float* scales,
    const bool* masks,
    const float* intrinsics,
    const int64_t* ii,
    const int64_t* jj,
    int n_edges,
    int height,
    int width,
    int stride,
    int offset,
    int sample_height,
    int sample_width,
    float* distances) {
    const int edge = blockIdx.x;
    if (edge >= n_edges) {
        return;
    }

    __shared__ float R_rel[9];
    __shared__ float t_rel[3];
    __shared__ float flow_sum_shared[THREADS];
    __shared__ float valid_count_shared[THREADS];
    __shared__ float source_count_shared[THREADS];

    const int source = static_cast<int>(ii[edge]);
    const int target = static_cast<int>(jj[edge]);
    if (threadIdx.x == 0) {
        const float* pose_i = poses + source * 7;
        const float* pose_j = poses + target * 7;
        float pred_t[3];
        float pred_q[4];
        relative_pose(pose_i, pose_j, pred_t, pred_q);
        quat_to_matrix(pred_q, R_rel);
        t_rel[0] = pred_t[0];
        t_rel[1] = pred_t[1];
        t_rel[2] = pred_t[2];
    }
    __syncthreads();

    const float fx = intrinsics[0];
    const float fy = intrinsics[1];
    const float cx = intrinsics[2];
    const float cy = intrinsics[3];
    const float scale = scales[source];
    const int total_samples = sample_height * sample_width;
    float flow_sum = 0.0f;
    float valid_count = 0.0f;
    float source_count = 0.0f;

    for (int sample = threadIdx.x; sample < total_samples; sample += blockDim.x) {
        const int sy = sample / sample_width;
        const int sx = sample - sy * sample_width;
        const int y = offset + sy * stride;
        const int x = offset + sx * stride;
        const int pixel = source * height * width + y * width + x;
        if (!masks[pixel]) {
            continue;
        }
        source_count += 1.0f;

        const float metric_depth = depths[pixel] * scale;
        const float disp = 1.0f / fmaxf(metric_depth, 1.0e-6f);
        const float X0 = (static_cast<float>(sx) - cx) / fx;
        const float Y0 = (static_cast<float>(sy) - cy) / fy;
        const float Z0 = 1.0f;

        const float X1 =
            R_rel[0] * X0 + R_rel[1] * Y0 + R_rel[2] * Z0 + t_rel[0] * disp;
        const float Y1 =
            R_rel[3] * X0 + R_rel[4] * Y0 + R_rel[5] * Z0 + t_rel[1] * disp;
        const float Z1 =
            R_rel[6] * X0 + R_rel[7] * Y0 + R_rel[8] * Z0 + t_rel[2] * disp;
        if (Z1 <= MIN_DEPTH) {
            continue;
        }

        const float inv_Z1 = 1.0f / Z1;
        const float px = fx * X1 * inv_Z1 + cx;
        const float py = fy * Y1 * inv_Z1 + cy;
        const float dx = px - static_cast<float>(sx);
        const float dy = py - static_cast<float>(sy);
        flow_sum += fminf(sqrtf(dx * dx + dy * dy), 256.0f);
        valid_count += 1.0f;
    }

    flow_sum_shared[threadIdx.x] = flow_sum;
    valid_count_shared[threadIdx.x] = valid_count;
    source_count_shared[threadIdx.x] = source_count;
    __syncthreads();

    for (int stride_reduce = blockDim.x / 2; stride_reduce > 0; stride_reduce >>= 1) {
        if (threadIdx.x < stride_reduce) {
            flow_sum_shared[threadIdx.x] += flow_sum_shared[threadIdx.x + stride_reduce];
            valid_count_shared[threadIdx.x] += valid_count_shared[threadIdx.x + stride_reduce];
            source_count_shared[threadIdx.x] += source_count_shared[threadIdx.x + stride_reduce];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const float total_flow = flow_sum_shared[0];
        const float total_valid = valid_count_shared[0];
        const float total_source = source_count_shared[0];
        const float valid_ratio = total_valid / fmaxf(total_source, 1.0f);
        distances[edge] = valid_ratio > 0.5f ? total_flow / fmaxf(total_valid, 1.0f) : INFINITY;
    }
}

}  // namespace

torch::Tensor projection_distance_cuda(
    torch::Tensor poses,
    torch::Tensor depths,
    torch::Tensor scales,
    torch::Tensor masks,
    torch::Tensor intrinsics,
    torch::Tensor ii,
    torch::Tensor jj,
    int64_t stride) {
    CHECK_CUDA(poses);
    CHECK_CUDA(depths);
    CHECK_CUDA(scales);
    CHECK_CUDA(masks);
    CHECK_CUDA(intrinsics);
    CHECK_FLOAT(poses);
    CHECK_FLOAT(depths);
    CHECK_FLOAT(scales);
    CHECK_BOOL(masks);
    CHECK_FLOAT(intrinsics);
    CHECK_CONTIGUOUS(poses);
    CHECK_CONTIGUOUS(depths);
    CHECK_CONTIGUOUS(scales);
    CHECK_CONTIGUOUS(masks);
    CHECK_CONTIGUOUS(intrinsics);
    TORCH_CHECK(poses.dim() == 2 && poses.size(1) == 7, "poses must have shape (N, 7)");
    TORCH_CHECK(depths.dim() == 3, "depths must have shape (N, H, W)");
    TORCH_CHECK(scales.dim() == 1 && scales.numel() == poses.size(0), "scales must have shape (N,)");
    TORCH_CHECK(masks.sizes() == depths.sizes(), "masks must match depths shape");
    TORCH_CHECK(intrinsics.dim() == 1 && intrinsics.numel() == 4, "intrinsics must have shape (4,)");
    const int64_t n_nodes64 = poses.size(0);
    const int64_t n_edges64 = ii.numel();
    TORCH_CHECK(depths.size(0) == n_nodes64, "depths node count mismatch");
    check_edge_index_tensors(ii, jj, n_edges64);
    TORCH_CHECK(stride > 0, "stride must be positive");
    const int64_t offset64 = stride / 2 - 1;
    TORCH_CHECK(offset64 >= 0, "stride must be at least 2 for centered projection sampling");
    const int64_t height64 = depths.size(1);
    const int64_t width64 = depths.size(2);
    TORCH_CHECK(offset64 < height64 && offset64 < width64, "stride offset is outside depth image");
    const int64_t sample_height64 = (height64 - offset64 + stride - 1) / stride;
    const int64_t sample_width64 = (width64 - offset64 + stride - 1) / stride;
    TORCH_CHECK(n_edges64 <= std::numeric_limits<int>::max(), "too many edges for geometry projection distance");
    TORCH_CHECK(height64 <= std::numeric_limits<int>::max(), "height too large for geometry projection distance");
    TORCH_CHECK(width64 <= std::numeric_limits<int>::max(), "width too large for geometry projection distance");

    const c10::cuda::CUDAGuard device_guard(poses.device());
    auto distances = torch::empty({n_edges64}, poses.options());
    const int n_edges = static_cast<int>(n_edges64);
    if (n_edges > 0) {
        projection_distance_kernel<<<n_edges, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
            poses.data_ptr<float>(),
            depths.data_ptr<float>(),
            scales.data_ptr<float>(),
            masks.data_ptr<bool>(),
            intrinsics.data_ptr<float>(),
            ii.data_ptr<int64_t>(),
            jj.data_ptr<int64_t>(),
            n_edges,
            static_cast<int>(height64),
            static_cast<int>(width64),
            static_cast<int>(stride),
            static_cast<int>(offset64),
            static_cast<int>(sample_height64),
            static_cast<int>(sample_width64),
            distances.data_ptr<float>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return distances;
}
