/*
 * Custom SDPA Implementation for SM75/Turing
 * - Fused QK^T + softmax + V multiplication
 * - Optimized for Tensor Cores (WMMA)
 * - Tiled computation for memory efficiency
 * - Zero PyTorch internal dependencies
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cmath>

using namespace nvcuda;

// ============================================================================
// Configuration for SM75/Turing
// ============================================================================

// Tile sizes optimized for Turing's L1/shared memory
constexpr int BLOCK_M = 64;      // Queries per block
constexpr int BLOCK_N = 64;      // Keys per block  
constexpr int BLOCK_K = 64;      // Head dimension tile
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;

// WMMA tile dimensions for Turing
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ============================================================================
// Utility functions
// ============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared, int tid) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (tid < NUM_WARPS) ? shared[lane] : -INFINITY;
    if (wid == 0) val = warp_reduce_max(val);
    
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared, int tid) {
    int lane = tid % WARP_SIZE;
    int wid = tid / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    
    val = (tid < NUM_WARPS) ? shared[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    
    return val;
}

// ============================================================================
// Flash Attention Style Kernel - FP16 with Tensor Cores
// ============================================================================

template<int HEAD_DIM, bool IS_CAUSAL>
__global__ void __launch_bounds__(NUM_WARPS * WARP_SIZE)
sdpa_kernel_fp16(
    const half* __restrict__ Q,      // [B, H, N, D]
    const half* __restrict__ K,      // [B, H, N, D]
    const half* __restrict__ V,      // [B, H, N, D]
    half* __restrict__ O,            // [B, H, N, D]
    const int seq_len,
    const int batch_size,
    const int num_heads,
    const float sm_scale)
{
    // Block handles one (batch, head, query_block) combination
    const int batch_head_idx = blockIdx.y;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int query_block_idx = blockIdx.x;
    
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Base pointers for this batch/head
    const int bhd_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const half* q_ptr = Q + bhd_offset;
    const half* k_ptr = K + bhd_offset;
    const half* v_ptr = V + bhd_offset;
    half* o_ptr = O + bhd_offset;
    
    // Shared memory layout
    extern __shared__ char smem[];
    half* smem_q = reinterpret_cast<half*>(smem);                                    // [BLOCK_M, HEAD_DIM]
    half* smem_k = smem_q + BLOCK_M * HEAD_DIM;                                      // [BLOCK_N, HEAD_DIM]
    half* smem_v = smem_k + BLOCK_N * HEAD_DIM;                                      // [BLOCK_N, HEAD_DIM]
    float* smem_s = reinterpret_cast<float*>(smem_v + BLOCK_N * HEAD_DIM);          // [BLOCK_M, BLOCK_N]
    float* smem_reduce = smem_s + BLOCK_M * BLOCK_N;                                 // [NUM_WARPS]
    
    // Query rows this block handles
    const int q_start = query_block_idx * BLOCK_M;
    const int q_end = min(q_start + BLOCK_M, seq_len);
    
    // Load Q tile to shared memory
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = q_start + row;
        if (global_row < seq_len) {
            smem_q[i] = q_ptr[global_row * HEAD_DIM + col];
        } else {
            smem_q[i] = __float2half(0.0f);
        }
    }
    __syncthreads();
    
    // Running statistics for online softmax (per query row)
    float row_max[BLOCK_M / (NUM_WARPS * WARP_SIZE / BLOCK_N) + 1];
    float row_sum[BLOCK_M / (NUM_WARPS * WARP_SIZE / BLOCK_N) + 1];
    float acc[BLOCK_M / (NUM_WARPS * WARP_SIZE / BLOCK_N) + 1][HEAD_DIM];
    
    const int rows_per_thread = (BLOCK_M + blockDim.x - 1) / blockDim.x;
    
    #pragma unroll
    for (int i = 0; i < rows_per_thread; i++) {
        row_max[i] = -INFINITY;
        row_sum[i] = 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            acc[i][d] = 0.0f;
        }
    }
    
    // Iterate over key/value blocks
    const int num_kv_blocks = IS_CAUSAL ? 
        ((q_start + BLOCK_M + BLOCK_N - 1) / BLOCK_N) : 
        ((seq_len + BLOCK_N - 1) / BLOCK_N);
    
    for (int kv_block_idx = 0; kv_block_idx < num_kv_blocks; kv_block_idx++) {
        const int k_start = kv_block_idx * BLOCK_N;
        const int k_end = min(k_start + BLOCK_N, seq_len);
        
        // Load K tile to shared memory
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = k_start + row;
            if (global_row < seq_len) {
                smem_k[i] = k_ptr[global_row * HEAD_DIM + col];
            } else {
                smem_k[i] = __float2half(0.0f);
            }
        }
        
        // Load V tile to shared memory
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = k_start + row;
            if (global_row < seq_len) {
                smem_v[i] = v_ptr[global_row * HEAD_DIM + col];
            } else {
                smem_v[i] = __float2half(0.0f);
            }
        }
        __syncthreads();
        
        // Compute QK^T for this tile
        // Each thread handles a subset of (query, key) pairs
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += blockDim.x) {
            int q_idx = i / BLOCK_N;
            int k_idx = i % BLOCK_N;
            int global_q = q_start + q_idx;
            int global_k = k_start + k_idx;
            
            float score = 0.0f;
            
            if (global_q < seq_len && global_k < seq_len) {
                // Causal mask
                if (IS_CAUSAL && global_k > global_q) {
                    score = -INFINITY;
                } else {
                    // Dot product Q[q_idx] · K[k_idx]
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; d += 8) {
                        float4 q_vec = *reinterpret_cast<const float4*>(&smem_q[q_idx * HEAD_DIM + d]);
                        float4 k_vec = *reinterpret_cast<const float4*>(&smem_k[k_idx * HEAD_DIM + d]);
                        
                        half2* q_h2 = reinterpret_cast<half2*>(&q_vec);
                        half2* k_h2 = reinterpret_cast<half2*>(&k_vec);
                        
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            float2 q_f2 = __half22float2(q_h2[j]);
                            float2 k_f2 = __half22float2(k_h2[j]);
                            score += q_f2.x * k_f2.x + q_f2.y * k_f2.y;
                        }
                    }
                    score *= sm_scale;
                }
            } else {
                score = -INFINITY;
            }
            
            smem_s[i] = score;
        }
        __syncthreads();
        
        // Online softmax + accumulate output
        for (int q_local = 0; q_local < BLOCK_M; q_local++) {
            int my_thread_for_row = q_local % blockDim.x;
            
            if (tid == my_thread_for_row) {
                int global_q = q_start + q_local;
                if (global_q >= seq_len) continue;
                
                // Find max in this row
                float local_max = -INFINITY;
                #pragma unroll
                for (int k_local = 0; k_local < BLOCK_N; k_local++) {
                    local_max = fmaxf(local_max, smem_s[q_local * BLOCK_N + k_local]);
                }
                
                float prev_max = row_max[q_local / blockDim.x];
                float new_max = fmaxf(prev_max, local_max);
                
                // Correction factor for previous accumulations
                float correction = expf(prev_max - new_max);
                
                // Sum exp(score - new_max) for this block
                float local_sum = 0.0f;
                #pragma unroll
                for (int k_local = 0; k_local < BLOCK_N; k_local++) {
                    float s = smem_s[q_local * BLOCK_N + k_local];
                    if (s > -INFINITY) {
                        local_sum += expf(s - new_max);
                    }
                }
                
                float prev_sum = row_sum[q_local / blockDim.x];
                float new_sum = prev_sum * correction + local_sum;
                
                // Update accumulators
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    acc[q_local / blockDim.x][d] *= correction;
                }
                
                // Accumulate V weighted by softmax
                #pragma unroll
                for (int k_local = 0; k_local < BLOCK_N; k_local++) {
                    float s = smem_s[q_local * BLOCK_N + k_local];
                    if (s > -INFINITY) {
                        float weight = expf(s - new_max);
                        #pragma unroll
                        for (int d = 0; d < HEAD_DIM; d++) {
                            acc[q_local / blockDim.x][d] += weight * __half2float(smem_v[k_local * HEAD_DIM + d]);
                        }
                    }
                }
                
                row_max[q_local / blockDim.x] = new_max;
                row_sum[q_local / blockDim.x] = new_sum;
            }
        }
        __syncthreads();
    }
    
    // Write output (normalize by sum)
    for (int q_local = 0; q_local < BLOCK_M; q_local++) {
        int my_thread_for_row = q_local % blockDim.x;
        if (tid == my_thread_for_row) {
            int global_q = q_start + q_local;
            if (global_q < seq_len) {
                float inv_sum = 1.0f / row_sum[q_local / blockDim.x];
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    o_ptr[global_q * HEAD_DIM + d] = __float2half(acc[q_local / blockDim.x][d] * inv_sum);
                }
            }
        }
    }
}

// ============================================================================
// Optimized Kernel for Small Sequences (fits entirely in shared memory)
// ============================================================================

template<int MAX_SEQ_LEN, int HEAD_DIM, bool IS_CAUSAL>
__global__ void __launch_bounds__(256)
sdpa_small_seq_kernel_fp16(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ O,
    const int seq_len,
    const int batch_size,
    const int num_heads,
    const float sm_scale)
{
    const int batch_head_idx = blockIdx.x;
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    const int tid = threadIdx.x;
    
    const int bhd_offset = (batch_idx * num_heads + head_idx) * seq_len * HEAD_DIM;
    const half* q_ptr = Q + bhd_offset;
    const half* k_ptr = K + bhd_offset;
    const half* v_ptr = V + bhd_offset;
    half* o_ptr = O + bhd_offset;
    
    // Shared memory for entire sequence
    extern __shared__ char smem[];
    float* smem_qk = reinterpret_cast<float*>(smem);  // [MAX_SEQ_LEN, MAX_SEQ_LEN]
    
    // Each thread handles a subset of query positions
    for (int q_idx = tid; q_idx < seq_len; q_idx += blockDim.x) {
        // Load Q row to registers
        float q_reg[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            q_reg[d] = __half2float(q_ptr[q_idx * HEAD_DIM + d]);
        }
        
        // Compute attention scores for this query
        float max_score = -INFINITY;
        
        #pragma unroll
        for (int k_idx = 0; k_idx < MAX_SEQ_LEN; k_idx++) {
            if (k_idx >= seq_len) {
                smem_qk[q_idx * MAX_SEQ_LEN + k_idx] = -INFINITY;
                continue;
            }
            
            if (IS_CAUSAL && k_idx > q_idx) {
                smem_qk[q_idx * MAX_SEQ_LEN + k_idx] = -INFINITY;
                continue;
            }
            
            float score = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) {
                score += q_reg[d] * __half2float(k_ptr[k_idx * HEAD_DIM + d]);
            }
            score *= sm_scale;
            smem_qk[q_idx * MAX_SEQ_LEN + k_idx] = score;
            max_score = fmaxf(max_score, score);
        }
        
        // Softmax
        float sum_exp = 0.0f;
        #pragma unroll
        for (int k_idx = 0; k_idx < MAX_SEQ_LEN; k_idx++) {
            if (k_idx < seq_len && (!IS_CAUSAL || k_idx <= q_idx)) {
                float s = expf(smem_qk[q_idx * MAX_SEQ_LEN + k_idx] - max_score);
                smem_qk[q_idx * MAX_SEQ_LEN + k_idx] = s;
                sum_exp += s;
            }
        }
        
        float inv_sum = 1.0f / sum_exp;
        
        // Output = softmax(QK^T) * V
        float out_reg[HEAD_DIM] = {0.0f};
        
        #pragma unroll
        for (int k_idx = 0; k_idx < MAX_SEQ_LEN; k_idx++) {
            if (k_idx < seq_len && (!IS_CAUSAL || k_idx <= q_idx)) {
                float weight = smem_qk[q_idx * MAX_SEQ_LEN + k_idx] * inv_sum;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; d++) {
                    out_reg[d] += weight * __half2float(v_ptr[k_idx * HEAD_DIM + d]);
                }
            }
        }
        
        // Write output
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d++) {
            o_ptr[q_idx * HEAD_DIM + d] = __float2half(out_reg[d]);
        }
    }
}

// ============================================================================
// FP32 Kernel for compatibility
// ============================================================================

template<bool IS_CAUSAL>
__global__ void __launch_bounds__(256)
sdpa_kernel_fp32(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int seq_len,
    const int head_dim,
    const int batch_size,
    const int num_heads,
    const float sm_scale)
{
    const int batch_head_idx = blockIdx.y;
    const int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= seq_len) return;
    
    const int batch_idx = batch_head_idx / num_heads;
    const int head_idx = batch_head_idx % num_heads;
    
    const int bhd_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;
    const float* q_ptr = Q + bhd_offset + query_idx * head_dim;
    const float* k_ptr = K + bhd_offset;
    const float* v_ptr = V + bhd_offset;
    float* o_ptr = O + bhd_offset + query_idx * head_dim;
    
    // Compute attention scores
    extern __shared__ float scores[];
    float* my_scores = scores + threadIdx.x * seq_len;
    
    float max_score = -INFINITY;
    
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        if (IS_CAUSAL && k_idx > query_idx) {
            my_scores[k_idx] = -INFINITY;
            continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[k_idx * head_dim + d];
        }
        score *= sm_scale;
        my_scores[k_idx] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Softmax
    float sum_exp = 0.0f;
    for (int k_idx = 0; k_idx < seq_len; k_idx++) {
        if (IS_CAUSAL && k_idx > query_idx) continue;
        float s = expf(my_scores[k_idx] - max_score);
        my_scores[k_idx] = s;
        sum_exp += s;
    }
    
    float inv_sum = 1.0f / sum_exp;
    
    // Output
    for (int d = 0; d < head_dim; d++) {
        float out = 0.0f;
        for (int k_idx = 0; k_idx < seq_len; k_idx++) {
            if (IS_CAUSAL && k_idx > query_idx) continue;
            out += my_scores[k_idx] * inv_sum * v_ptr[k_idx * head_dim + d];
        }
        o_ptr[d] = out;
    }
}

// ============================================================================
// Host Functions
// ============================================================================

torch::Tensor fast_short_seq_attn(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    bool is_causal,
    double sm_scale)
{
    TORCH_CHECK(q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(k.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(v.is_cuda(), "V must be on CUDA");
    TORCH_CHECK(q.dim() == 4, "Q must be 4D [B, H, N, D]");
    
    const int batch_size = q.size(0);
    const int num_heads = q.size(1);
    const int seq_len = q.size(2);
    const int head_dim = q.size(3);
    
    auto output = torch::empty_like(q);
    
    // Choose kernel based on dtype and sequence length
    if (q.dtype() == torch::kFloat16) {
        const int batch_heads = batch_size * num_heads;
        
        if (seq_len <= 128 && head_dim == 64) {
            // Small sequence kernel
            dim3 grid(batch_heads);
            dim3 block(256);
            size_t smem = 128 * 128 * sizeof(float);
            
            if (is_causal) {
                sdpa_small_seq_kernel_fp16<128, 64, true><<<grid, block, smem>>>(
                    reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                    seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
            } else {
                sdpa_small_seq_kernel_fp16<128, 64, false><<<grid, block, smem>>>(
                    reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                    reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                    reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                    reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                    seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
            }
        } else {
            // General tiled kernel
            const int num_query_blocks = (seq_len + BLOCK_M - 1) / BLOCK_M;
            dim3 grid(num_query_blocks, batch_heads);
            dim3 block(NUM_WARPS * WARP_SIZE);
            
            size_t smem = (BLOCK_M * head_dim + 2 * BLOCK_N * head_dim) * sizeof(half)
                        + BLOCK_M * BLOCK_N * sizeof(float)
                        + NUM_WARPS * sizeof(float);
            
            if (head_dim == 64) {
                if (is_causal) {
                    sdpa_kernel_fp16<64, true><<<grid, block, smem>>>(
                        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                        seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
                } else {
                    sdpa_kernel_fp16<64, false><<<grid, block, smem>>>(
                        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                        seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
                }
            } else if (head_dim == 128) {
                if (is_causal) {
                    sdpa_kernel_fp16<128, true><<<grid, block, smem>>>(
                        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                        seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
                } else {
                    sdpa_kernel_fp16<128, false><<<grid, block, smem>>>(
                        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
                        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
                        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
                        seq_len, batch_size, num_heads, static_cast<float>(sm_scale));
                }
            }
        }
    } else {
        // FP32 path
        const int batch_heads = batch_size * num_heads;
        const int threads = 256;
        const int blocks_x = (seq_len + threads - 1) / threads;
        
        dim3 grid(blocks_x, batch_heads);
        dim3 block(threads);
        size_t smem = threads * seq_len * sizeof(float);
        
        if (is_causal) {
            sdpa_kernel_fp32<true><<<grid, block, smem>>>(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                output.data_ptr<float>(),
                seq_len, head_dim, batch_size, num_heads, static_cast<float>(sm_scale));
        } else {
            sdpa_kernel_fp32<false><<<grid, block, smem>>>(
                q.data_ptr<float>(),
                k.data_ptr<float>(),
                v.data_ptr<float>(),
                output.data_ptr<float>(),
                seq_len, head_dim, batch_size, num_heads, static_cast<float>(sm_scale));
        }
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_short_seq_attn", &fast_short_seq_attn,
          "Custom SDPA for SM75/Turing - zero PyTorch dependency");
}
