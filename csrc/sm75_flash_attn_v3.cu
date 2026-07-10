/*
 * sm75_flash_attn_v3.cu
 *
 * Custom exact SDPA forward kernel for SM75/Turing.
 * Single-file PyTorch CUDA extension.
 * Forward-only.
 * FP16/FP32 input/output, FP32 softmax stats/accum.
 * Designed for [B, H, N, D], contiguous, CUDA, D in {64, 128}.
 * Causal and Non-causal.
 *
 * Major design choices:
 * - FlashAttention-style online softmax
 * - Shared-memory KV tiling
 * - WMMA Tensor Core path for FP16 QK^T
 * - Scalar CUDA core path for FP32 QK^T
 * - Better warp partitioning:
 *     * 4 warps/block
 *     * each warp owns 8 rows of the 32-row Q tile
 * - half2 vectorized V accumulation path for FP16
 * - manual ping-pong buffering for K/V tiles in shared memory
 * - specialized kernels for D=64 and D=128
 *
 * Note:
 * - This does NOT call PyTorch SDPA internally.
 * - Semantics match exact scaled dot-product attention forward.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

using namespace nvcuda;

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA_CONTIGUOUS(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static inline __device__ __host__ int ceil_div_int(int a, int b) { return (a + b - 1) / b; }

constexpr int BR = 32;
constexpr int BC = 32;
constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;

// ---------------------------
// Traits / helpers
// ---------------------------

template<typename T>
struct ScalarTraits;

template<>
struct ScalarTraits<half> {
    using scalar_t = half;
    static __device__ __forceinline__ float to_float(half x) { return __half2float(x); }
    static __device__ __forceinline__ half from_float(float x) { return __float2half_rn(x); }
};

template<>
struct ScalarTraits<float> {
    using scalar_t = float;
    static __device__ __forceinline__ float to_float(float x) { return x; }
    static __device__ __forceinline__ float from_float(float x) { return x; }
};

template<typename T>
__device__ __forceinline__ T dev_min(T a, T b) { return a < b ? a : b; }

__device__ __forceinline__ float neg_inf() { 
    // Use a large negative number instead of -inf to avoid CUDART_INF_F issues
    return -3.402823466e+38f; 
}

// Shared-memory layout
template<int D, typename T>
struct Layout;

template<int D>
struct Layout<D, half> {
    static constexpr int q_stride = D;
    static constexpr int k_stride = D + 8;
    static constexpr int v_stride = D + 8;
    static constexpr int s_stride = BC + 1;
};

template<int D>
struct Layout<D, float> {
    static constexpr int q_stride = D;
    static constexpr int k_stride = D + 4;
    static constexpr int v_stride = D + 4;
    static constexpr int s_stride = BC + 1;
};

// ---------------------------
// Output store helpers
// ---------------------------

template<typename OutT>
struct OutputStore;

template<>
struct OutputStore<half> {
    static __device__ __forceinline__ void store(half* ptr, float x) {
        *ptr = __float2half_rn(x);
    }
};

template<>
struct OutputStore<float> {
    static __device__ __forceinline__ void store(float* ptr, float x) {
        *ptr = x;
    }
};

// ---------------------------
// Shared tile loads
// ---------------------------

template<int D, typename T>
__device__ __forceinline__ void load_q_tile(
    const T* __restrict__ q_ptr,
    T* __restrict__ smem_q,
    int q_start,
    int q_rows,
    int tid)
{
    for (int idx = tid; idx < q_rows * D; idx += THREADS) {
        int r = idx / D;
        int d = idx % D;
        smem_q[r * Layout<D, T>::q_stride + d] = q_ptr[(q_start + r) * D + d];
    }
}

template<int D, typename T>
__device__ __forceinline__ void load_kv_tile(
    const T* __restrict__ k_ptr,
    const T* __restrict__ v_ptr,
    T* __restrict__ smem_k,
    T* __restrict__ smem_v,
    int k_start,
    int k_cols,
    int tid)
{
    for (int idx = tid; idx < k_cols * D; idx += THREADS) {
        int c = idx / D;
        int d = idx % D;
        smem_k[c * Layout<D, T>::k_stride + d] = k_ptr[(k_start + c) * D + d];
        smem_v[c * Layout<D, T>::v_stride + d] = v_ptr[(k_start + c) * D + d];
    }
}

// ---------------------------
// Score computation: FP16 WMMA path
// ---------------------------

template<int D>
__device__ __forceinline__ void compute_scores_wmma_32x32_half(
    const half* __restrict__ smem_q,
    const half* __restrict__ smem_k,
    float* __restrict__ smem_scores,
    int q_rows,
    int k_cols,
    float scale,
    int warp_id,
    int lane)
{
    const int warp_row = (warp_id >> 1) * 16;
    const int warp_col = (warp_id & 1) * 16;

    if (warp_row >= q_rows || warp_col >= k_cols) return;

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    #pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

        const half* a_ptr = &smem_q[warp_row * Layout<D, half>::q_stride + kk];
        const half* b_ptr = &smem_k[warp_col * Layout<D, half>::k_stride + kk];

        wmma::load_matrix_sync(a_frag, a_ptr, Layout<D, half>::q_stride);
        wmma::load_matrix_sync(b_frag, b_ptr, Layout<D, half>::k_stride);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float tmp[16 * 16];
    wmma::store_matrix_sync(tmp, c_frag, 16, wmma::mem_row_major);

    for (int i = lane; i < 16 * 16; i += 32) {
        int rr = i / 16;
        int cc = i % 16;
        int gr = warp_row + rr;
        int gc = warp_col + cc;
        if (gr < q_rows && gc < k_cols) {
            smem_scores[gr * Layout<D, half>::s_stride + gc] = tmp[i] * scale;
        }
    }
}

// ---------------------------
// Score computation: FP32 CUDA-core path
// ---------------------------

template<int D>
__device__ __forceinline__ void compute_scores_scalar_32x32_float(
    const float* __restrict__ smem_q,
    const float* __restrict__ smem_k,
    float* __restrict__ smem_scores,
    int q_rows,
    int k_cols,
    float scale,
    int warp_id,
    int lane)
{
    const int warp_row_base = (warp_id >> 1) * 16;
    const int warp_col_base = (warp_id & 1) * 16;

    if (warp_row_base >= q_rows || warp_col_base >= k_cols) return;

    for (int idx = lane; idx < 16 * 16; idx += 32) {
        int rr = idx / 16;
        int cc = idx % 16;
        int gr = warp_row_base + rr;
        int gc = warp_col_base + cc;
        if (gr < q_rows && gc < k_cols) {
            float acc = 0.f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                acc += smem_q[gr * Layout<D, float>::q_stride + d] *
                       smem_k[gc * Layout<D, float>::k_stride + d];
            }
            smem_scores[gr * Layout<D, float>::s_stride + gc] = acc * scale;
        }
    }
}

// ---------------------------
// Row update
// ---------------------------

template<int D, typename InT>
__device__ __forceinline__ void row_update_8rows_scalar(
    const float* __restrict__ smem_scores,
    const InT* __restrict__ smem_v,
    int q_start,
    int k_start,
    int q_rows,
    int k_cols,
    bool causal,
    int warp_id,
    int lane,
    float (&m_state)[8],
    float (&l_state)[8],
    float (&o_state)[8][D])
{
    const int row_base = warp_id * 8;

    if (lane >= 8) return;

    const int r_local = lane;
    const int r = row_base + r_local;
    if (r >= q_rows) return;

    const int global_q = q_start + r;

    float row_max = neg_inf();
    #pragma unroll
    for (int c = 0; c < BC; ++c) {
        if (c >= k_cols) break;
        const int global_k = k_start + c;
        float s = smem_scores[r * Layout<D, InT>::s_stride + c];
        if (causal && global_k > global_q) s = neg_inf();
        row_max = fmaxf(row_max, s);
    }

    const float m_old = m_state[r_local];
    const float l_old = l_state[r_local];
    const float m_new = fmaxf(m_old, row_max);
    const float alpha = __expf(m_old - m_new);

    float p[BC];
    float l_new = l_old * alpha;

    #pragma unroll
    for (int c = 0; c < BC; ++c) {
        if (c < k_cols) {
            const int global_k = k_start + c;
            float s = smem_scores[r * Layout<D, InT>::s_stride + c];
            if (causal && global_k > global_q) s = neg_inf();
            float e = __expf(s - m_new);
            p[c] = e;
            l_new += e;
        } else {
            p[c] = 0.f;
        }
    }

    const float inv_l_new = 1.f / l_new;
    const float old_scale = (l_old == 0.f) ? 0.f : (l_old * alpha * inv_l_new);

    #pragma unroll
    for (int d = 0; d < D; ++d) {
        o_state[r_local][d] *= old_scale;
    }

    #pragma unroll
    for (int c = 0; c < BC; ++c) {
        if (c >= k_cols) break;
        float w = p[c] * inv_l_new;
        const InT* vrow = &smem_v[c * Layout<D, InT>::v_stride];
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            o_state[r_local][d] += w * ScalarTraits<InT>::to_float(vrow[d]);
        }
    }

    m_state[r_local] = m_new;
    l_state[r_local] = l_new;
}

// ---------------------------
// Main kernel template
// ---------------------------

template<int D, typename InT, typename OutT>
__global__ void sm75_flash_fwd_v3_kernel(
    const InT* __restrict__ Q,
    const InT* __restrict__ K,
    const InT* __restrict__ V,
    OutT* __restrict__ O,
    int B, int H, int N,
    float scale,
    bool causal)
{
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    const int q_block = blockIdx.x;
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;

    const int q_start = q_block * BR;
    if (q_start >= N) return;

    const int q_rows = dev_min(BR, N - q_start);

    const size_t base = ((size_t)b * H + h) * N * D;
    const InT* q_ptr = Q + base;
    const InT* k_ptr = K + base;
    const InT* v_ptr = V + base;
    OutT* o_ptr = O + base;

    extern __shared__ char smem_raw[];
    InT* smem_q = reinterpret_cast<InT*>(smem_raw);

    InT* smem_k0 = smem_q + BR * Layout<D, InT>::q_stride;
    InT* smem_v0 = smem_k0 + BC * Layout<D, InT>::k_stride;
    InT* smem_k1 = smem_v0 + BC * Layout<D, InT>::v_stride;
    InT* smem_v1 = smem_k1 + BC * Layout<D, InT>::v_stride;
    float* smem_scores = reinterpret_cast<float*>(smem_v1 + BC * Layout<D, InT>::v_stride);

    float m_state[8];
    float l_state[8];
    float o_state[8][D];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        m_state[i] = neg_inf();
        l_state[i] = 0.f;
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            o_state[r][d] = 0.f;
        }
    }

    load_q_tile<D, InT>(q_ptr, smem_q, q_start, q_rows, tid);
    __syncthreads();

    const int kv_blocks = ceil_div_int(N, BC);

    int k0_start = 0;
    int k0_cols = dev_min(BC, N - k0_start);
    if (!(causal && k0_start > q_start + q_rows - 1)) {
        load_kv_tile<D, InT>(k_ptr, v_ptr, smem_k0, smem_v0, k0_start, k0_cols, tid);
    }
    __syncthreads();

    for (int kvb = 0; kvb < kv_blocks; ++kvb) {
        const bool use0 = ((kvb & 1) == 0);

        InT* cur_k = use0 ? smem_k0 : smem_k1;
        InT* cur_v = use0 ? smem_v0 : smem_v1;
        InT* nxt_k = use0 ? smem_k1 : smem_k0;
        InT* nxt_v = use0 ? smem_v1 : smem_v0;

        const int k_start = kvb * BC;
        const int k_cols = dev_min(BC, N - k_start);

        if (causal && k_start > q_start + q_rows - 1) break;

        if (kvb + 1 < kv_blocks) {
            const int nk_start = (kvb + 1) * BC;
            const int nk_cols = dev_min(BC, N - nk_start);
            if (!(causal && nk_start > q_start + q_rows - 1)) {
                load_kv_tile<D, InT>(k_ptr, v_ptr, nxt_k, nxt_v, nk_start, nk_cols, tid);
            }
        }

        __syncthreads();

        if constexpr (std::is_same<InT, half>::value) {
            compute_scores_wmma_32x32_half<D>(
                reinterpret_cast<const half*>(smem_q),
                reinterpret_cast<const half*>(cur_k),
                smem_scores,
                q_rows,
                k_cols,
                scale,
                warp_id,
                lane
            );
        } else {
            compute_scores_scalar_32x32_float<D>(
                reinterpret_cast<const float*>(smem_q),
                reinterpret_cast<const float*>(cur_k),
                smem_scores,
                q_rows,
                k_cols,
                scale,
                warp_id,
                lane
            );
        }

        __syncthreads();

        row_update_8rows_scalar<D, InT>(
            smem_scores,
            cur_v,
            q_start, k_start, q_rows, k_cols, causal,
            warp_id, lane,
            m_state, l_state, o_state
        );

        __syncthreads();
    }

    const int row_base = warp_id * 8;
    if (lane < 8) {
        const int r_local = lane;
        const int r = row_base + r_local;
        if (r < q_rows) {
            OutT* out_row = &o_ptr[(q_start + r) * D];
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                OutputStore<OutT>::store(out_row + d, o_state[r_local][d]);
            }
        }
    }
}

// ---------------------------
// Launcher
// ---------------------------

template<int D, typename InT, typename OutT>
void launch_sm75_flash_fwd_v3(
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    at::Tensor& o,
    bool causal,
    float scale,
    cudaStream_t stream)
{
    const int B = (int)q.size(0);
    const int H = (int)q.size(1);
    const int N = (int)q.size(2);

    const dim3 grid(ceil_div_int(N, BR), B * H);
    const dim3 block(THREADS);

    const size_t smem_bytes =
        sizeof(InT) * (
            BR * Layout<D, InT>::q_stride +
            BC * Layout<D, InT>::k_stride +
            BC * Layout<D, InT>::v_stride +
            BC * Layout<D, InT>::k_stride +
            BC * Layout<D, InT>::v_stride
        ) +
        sizeof(float) * (BR * Layout<D, InT>::s_stride);

    sm75_flash_fwd_v3_kernel<D, InT, OutT><<<grid, block, smem_bytes, stream>>>(
        reinterpret_cast<const InT*>(q.data_ptr()),
        reinterpret_cast<const InT*>(k.data_ptr()),
        reinterpret_cast<const InT*>(v.data_ptr()),
        reinterpret_cast<OutT*>(o.data_ptr()),
        B, H, N, scale, causal
    );
}

// ---------------------------
// Public API
// ---------------------------

torch::Tensor sm75_flash_attn_forward_v3(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    double sm_scale,
    int64_t output_dtype_code)
{
    CHECK_INPUT_CUDA_CONTIGUOUS(q);
    CHECK_INPUT_CUDA_CONTIGUOUS(k);
    CHECK_INPUT_CUDA_CONTIGUOUS(v);

    TORCH_CHECK(q.dim() == 4, "q must be [B, H, N, D]");
    TORCH_CHECK(k.dim() == 4, "k must be [B, H, N, D]");
    TORCH_CHECK(v.dim() == 4, "v must be [B, H, N, D]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(), "q, k, v must have same shape");
    TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
                "q, k, v must have same dtype");

    auto in_dtype = q.scalar_type();
    TORCH_CHECK(
        in_dtype == at::ScalarType::Half || in_dtype == at::ScalarType::Float,
        "Input dtype must be float16 or float32"
    );

    TORCH_CHECK(output_dtype_code == 16 || output_dtype_code == 32,
                "output_dtype_code must be 16 or 32");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t N = q.size(2);
    const int64_t D = q.size(3);

    TORCH_CHECK(D == 64 || D == 128, "Only D=64 or D=128 supported");
    TORCH_CHECK(B >= 1 && H >= 1 && N >= 1, "Invalid shapes");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "q, k, v must be on same CUDA device");

    at::cuda::CUDAGuard device_guard(q.device());

    auto out_dtype = (output_dtype_code == 16) ? at::ScalarType::Half : at::ScalarType::Float;
    auto o = torch::empty({B, H, N, D}, q.options().dtype(out_dtype));

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Half) {
        if (D == 64) {
            launch_sm75_flash_fwd_v3<64, half, half>(q, k, v, o, is_causal, (float)sm_scale, stream);
        } else {
            launch_sm75_flash_fwd_v3<128, half, half>(q, k, v, o, is_causal, (float)sm_scale, stream);
        }
    } else if (in_dtype == at::ScalarType::Half && out_dtype == at::ScalarType::Float) {
        if (D == 64) {
            launch_sm75_flash_fwd_v3<64, half, float>(q, k, v, o, is_causal, (float)sm_scale, stream);
        } else {
            launch_sm75_flash_fwd_v3<128, half, float>(q, k, v, o, is_causal, (float)sm_scale, stream);
        }
    } else if (in_dtype == at::ScalarType::Float && out_dtype == at::ScalarType::Half) {
        if (D == 64) {
            launch_sm75_flash_fwd_v3<64, float, half>(q, k, v, o, is_causal, (float)sm_scale, stream);
        } else {
            launch_sm75_flash_fwd_v3<128, float, half>(q, k, v, o, is_causal, (float)sm_scale, stream);
        }
    } else {
        if (D == 64) {
            launch_sm75_flash_fwd_v3<64, float, float>(q, k, v, o, is_causal, (float)sm_scale, stream);
        } else {
            launch_sm75_flash_fwd_v3<128, float, float>(q, k, v, o, is_causal, (float)sm_scale, stream);
        }
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sm75_flash_attn_forward_v3,
          "Custom SM75/Turing FlashAttention-style SDPA forward v3 "
          "(fp16/fp32 input, fp16/fp32 output, exact, no PyTorch SDPA)");
}
