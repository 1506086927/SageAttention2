#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

template<int D, bool IS_CAUSAL>
__global__ void __launch_bounds__(256)
fast_attn_kernel_half2_v2(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    int NQ,
    int NK,
    float scale,
    int num_heads,
    int num_kv_heads,
    // strides for Q
    int q_stride_b, int q_stride_h, int q_stride_seq,
    // strides for K
    int k_stride_b, int k_stride_h, int k_stride_seq,
    // strides for V
    int v_stride_b, int v_stride_h, int v_stride_seq,
    // strides for O
    int o_stride_b, int o_stride_h, int o_stride_seq
) {
    const int bh = blockIdx.x;  // batch * num_heads + head
    const int batch_idx = bh / num_heads;
    const int head_idx = bh % num_heads;
    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int b_kv_h = batch_idx * num_kv_heads + kv_head_idx;
    
    const int tid = threadIdx.x;
    
    // Shared memory layout: [smem_q, smem_k, smem_v, smem_s]
    extern __shared__ char smem_raw[];
    half* smem_q = reinterpret_cast<half*>(smem_raw);
    half* smem_k = smem_q + NQ * D;
    half* smem_v = smem_k + NK * D;
    float* smem_s = reinterpret_cast<float*>(smem_v + NK * D);

    // Load Q, K, V into shared memory using vectorized loads based on strides
    const int total_q_elements = NQ * D;
    const int total_kv_elements = NK * D;
    const int num_half2_q = total_q_elements / 2;
    const int num_half2_kv = total_kv_elements / 2;

    for (int idx = tid; idx < num_half2_q; idx += blockDim.x) {
        int base_idx = idx * 2;
        int r = base_idx / D;
        int d = base_idx % D;
        int offset_q = batch_idx * q_stride_b + head_idx * q_stride_h + r * q_stride_seq + d;
        reinterpret_cast<half2*>(&smem_q[r * D + d])[0] = reinterpret_cast<const half2*>(&Q[offset_q])[0];
    }

    for (int idx = tid; idx < num_half2_kv; idx += blockDim.x) {
        int base_idx = idx * 2;
        int r = base_idx / D;
        int d = base_idx % D;
        int offset_k = batch_idx * k_stride_b + kv_head_idx * k_stride_h + r * k_stride_seq + d;
        int offset_v = batch_idx * v_stride_b + kv_head_idx * v_stride_h + r * v_stride_seq + d;
        reinterpret_cast<half2*>(&smem_k[r * D + d])[0] = reinterpret_cast<const half2*>(&K[offset_k])[0];
        reinterpret_cast<half2*>(&smem_v[r * D + d])[0] = reinterpret_cast<const half2*>(&V[offset_v])[0];
    }
    __syncthreads();

    // Each thread computes one or more output rows for Q
    const int rows_per_thread = (NQ + blockDim.x - 1) / blockDim.x;
    const int row_start = min(tid * rows_per_thread, NQ);
    const int row_end = min(row_start + rows_per_thread, NQ);

    for (int i = row_start; i < row_end; ++i) {
        const int q_offset = i * D;
        
        // Compute Q[i] @ K^T -> scores
        float max_val = -1e20f;

        #pragma unroll
        for (int j = 0; j < NK; ++j) {
            if ((IS_CAUSAL && j > i) || (j >= NK) || (i >= NQ)) {
                smem_s[i * NK + j] = -1e20f;
                continue;
            }

            float score = 0.f;
            
            #pragma unroll
            for (int d = 0; d < D; d += 2) {
                half2 qh = reinterpret_cast<const half2*>(&smem_q[q_offset + d])[0];
                half2 kh = reinterpret_cast<const half2*>(&smem_k[j * D + d])[0];
                float2 qf = __half22float2(qh);
                float2 kf = __half22float2(kh);
                score += qf.x * kf.x + qf.y * kf.y;
            }
            score *= scale;
            smem_s[i * NK + j] = score;
            max_val = fmaxf(max_val, score);
        }

        // Softmax - compute in registers first
        float sum_exp = 0.f;
        #pragma unroll
        for (int j = 0; j < NK; ++j) {
            float s = __expf(smem_s[i * NK + j] - max_val);
            smem_s[i * NK + j] = s;
            sum_exp += s;
        }
        float inv_sum = __fdividef(1.f, fmaxf(sum_exp, 1e-20f));

        // PV: O[i] = sum_j(softmax[i,j] * V[j]) using vectorized stores
        #pragma unroll
        for (int d = 0; d < D; d += 2) {
            float acc0 = 0.f, acc1 = 0.f;
            #pragma unroll
            for (int j = 0; j < NK; ++j) {
                float w = smem_s[i * NK + j] * inv_sum;
                half2 vh = reinterpret_cast<const half2*>(&smem_v[j * D + d])[0];
                float2 vf = __half22float2(vh);
                acc0 += w * vf.x;
                acc1 += w * vf.y;
            }
            half2 oh = __floats2half2_rn(acc0, acc1);
            int o_global_offset = batch_idx * o_stride_b + head_idx * o_stride_h + i * o_stride_seq + d;
            reinterpret_cast<half2*>(&O[o_global_offset])[0] = oh;
        }
    }
}

at::Tensor sm75_short_sdpa_v2(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    bool is_causal, double sm_scale, std::string layout /* "HND" or "NHD" */
) {
    // P0-A-4: 短核入口防御性检查
    TORCH_CHECK(q.scalar_type() == at::kHalf, "sm75_short_sdpa_v2 requires fp16 q, got ", q.scalar_type());
    TORCH_CHECK(k.scalar_type() == at::kHalf, "sm75_short_sdpa_v2 requires fp16 k, got ", k.scalar_type());
    TORCH_CHECK(v.scalar_type() == at::kHalf, "sm75_short_sdpa_v2 requires fp16 v, got ", v.scalar_type());
    TORCH_CHECK(q.is_contiguous() || q.stride(-1) == 1, "q last dim must be contiguous for sm75 short v2 kernel");
    TORCH_CHECK(k.is_contiguous() || k.stride(-1) == 1, "k last dim must be contiguous for sm75 short v2 kernel");
    TORCH_CHECK(v.is_contiguous() || v.stride(-1) == 1, "v last dim must be contiguous for sm75 short v2 kernel");

    int seq_dim = (layout == "HND") ? 2 : 1;
    int NQ = q.size(seq_dim);
    int NK = k.size(seq_dim);
    int D = q.size(3);
    int batch_size = q.size(0);
    int num_heads = (layout == "HND") ? q.size(1) : q.size(2);
    int num_kv_heads = (layout == "HND") ? k.size(1) : k.size(2);

    TORCH_CHECK(num_heads % num_kv_heads == 0, "q_heads must be a multiple of kv_heads (GQA/MQA), got ", num_heads, " and ", num_kv_heads);

    // head_dim not in {64,128} 时的处理：返回未定义/不支持的哨兵（空 tensor）
    if (D != 64 && D != 128) {
        return torch::empty({0}, q.options());
    }

    size_t smem_bytes =
        size_t(NQ) * D * sizeof(half) +
        size_t(NK) * D * sizeof(half) +
        size_t(NK) * D * sizeof(half) +
        size_t(NQ) * size_t(NK) * sizeof(float);

    const int max_smem_per_block = 64 * 1024;
    if (smem_bytes > max_smem_per_block) {
        return torch::empty({0}, q.options()); // sentinel empty tensor for smem超限 fallback
    }

    auto output = torch::empty_like(q);

    const int total_bh = batch_size * num_heads;
    const int threads = 256;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // strides
    int q_stride_b = q.stride(0), q_stride_h = q.stride(1), q_stride_seq = q.stride(2);
    int k_stride_b = k.stride(0), k_stride_h = k.stride(1), k_stride_seq = k.stride(2);
    int v_stride_b = v.stride(0), v_stride_h = v.stride(1), v_stride_seq = v.stride(2);
    int o_stride_b = output.stride(0), o_stride_h = output.stride(1), o_stride_seq = output.stride(2);

    if (layout == "NHD") {
        q_stride_h = q.stride(2); q_stride_seq = q.stride(1);
        k_stride_h = k.stride(2); k_stride_seq = k.stride(1);
        v_stride_h = v.stride(2); v_stride_seq = v.stride(1);
        o_stride_h = output.stride(2); o_stride_seq = output.stride(1);
    }

    const half* q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(k.data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(v.data_ptr<at::Half>());
    half* o_ptr = reinterpret_cast<half*>(output.data_ptr<at::Half>());

    #define LAUNCH_CUSTOM_KERNEL_V2(D_val, causal) \
        cudaFuncSetAttribute(fast_attn_kernel_half2_v2<D_val, causal>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); \
        fast_attn_kernel_half2_v2<D_val, causal><<<total_bh, threads, smem_bytes, stream>>>( \
            q_ptr, k_ptr, v_ptr, o_ptr, NQ, NK, float(sm_scale), num_heads, num_kv_heads, \
            q_stride_b, q_stride_h, q_stride_seq, \
            k_stride_b, k_stride_h, k_stride_seq, \
            v_stride_b, v_stride_h, v_stride_seq, \
            o_stride_b, o_stride_h, o_stride_seq)

    if (D == 64) {
        if (is_causal) {
            LAUNCH_CUSTOM_KERNEL_V2(64, true);
        } else {
            LAUNCH_CUSTOM_KERNEL_V2(64, false);
        }
    } else if (D == 128) {
        if (is_causal) {
            LAUNCH_CUSTOM_KERNEL_V2(128, true);
        } else {
            LAUNCH_CUSTOM_KERNEL_V2(128, false);
        }
    }

    #undef LAUNCH_CUSTOM_KERNEL_V2

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return torch::empty({0}, q.options());
    }

    return output;
}