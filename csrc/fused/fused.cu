/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <ATen/cuda/CUDAContext.h>
 #include <torch/types.h>
 
 #include "../dispatch_utils.h"
 #include "../utils.cuh"
 #include "../reduction_utils.cuh"
 #include "../numeric_conversion.cuh"
 #include "../cp_async.cuh"
 #include <cuda_fp16.h>
 #include <cuda_bf16.h>
 
 enum class QuantType
 {
   kInt8,
   kInt4,
 };
 
 template <typename T>
 __device__ __forceinline__ float convert_to_float(T val)
 {
   if constexpr (std::is_same<T, half>::value)
   {
     return __half2float(val);
   }
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 800
   else if constexpr (std::is_same<T, nv_bfloat16>::value)
   {
     return __bfloat162float(val);
   }
 #endif
   else if constexpr (std::is_same<T, float>::value)
   {
     return val;
   }
 }
 
 template <typename T>
 __device__ __forceinline__ T convert_from_float(float val)
 {
   if constexpr (std::is_same<T, half>::value)
   {
     return __float2half_rn(val);
   }
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 800
   else if constexpr (std::is_same<T, nv_bfloat16>::value)
   {
     return __float2bfloat16_rn(val);
   }
 #endif
   else if constexpr (std::is_same<T, float>::value)
   {
     return val;
   }
 }
 
 template <typename T>
 struct PackTraits;
 
 template <>
 struct PackTraits<half>
 {
   static constexpr uint32_t pack_size = 8;
   static constexpr uint32_t elements_per_float4 = 8;
 };
 
 template <>
 struct PackTraits<nv_bfloat16>
 {
   static constexpr uint32_t pack_size = 8;
   static constexpr uint32_t elements_per_float4 = 8;
 };
 
 template <>
 struct PackTraits<float>
 {
   static constexpr uint32_t pack_size = 4;
   static constexpr uint32_t elements_per_float4 = 4;
 };
 
 template <typename T>
 struct KernelLaunchConfig {
   static constexpr uint32_t pack_size = PackTraits<T>::pack_size;
   
   static __host__ __forceinline__ 
   uint32_t calc_num_pack(uint32_t block_size, uint32_t head_dim) {
     return (block_size * (head_dim / pack_size) + 1023) / 1024;
   }
   
   static __host__ __forceinline__ 
   uint32_t calc_block_dim(uint32_t block_size, uint32_t head_dim, uint32_t num_pack) {
     return block_size * (head_dim / pack_size) / num_pack;
   }
 };
 
 template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, bool has_sm_scale = false, bool sub_mean = false, typename T>
 __global__ void QuantInt8Kernel(T *__restrict__ input, T *__restrict__ mean, int8_t *__restrict__ output, float *__restrict__ scale, float sm_scale, const uint32_t num_tokens,
                             const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                             const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                             const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output,
                             const uint32_t stride_bz_scale, const uint32_t stride_h_scale, uint32_t actual_head_dim)
 {
   static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");
 
   constexpr uint32_t pack_size = PackTraits<T>::pack_size;
   constexpr uint32_t num_threads_per_token = head_dim / pack_size;
 
   static_assert(num_threads_per_token <= 64, "The number of threads per token must be less than or equal to warp size");
 
   T x_val[num_pack_per_thread][pack_size];
   T mean_val[pack_size];
   float x_val_float[num_pack_per_thread][pack_size];
   float mean_val_float[pack_size];
 
   uint32_t bx = blockIdx.x;
   uint32_t head_id = blockIdx.y;
   uint32_t batch_id = blockIdx.z;
   uint32_t thread_id = threadIdx.x;
 
   uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
   T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
   T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
   int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
   float *scale_ptr_base = scale + batch_id * stride_bz_scale + head_id * stride_h_scale + bx;
 
   if constexpr (sub_mean)
   {
     *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
 #pragma unroll
     for (uint32_t j = 0; j < pack_size; j++)
     {
       mean_val_float[j] = convert_to_float(mean_val[j]);
     }
   }
 
   constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;
 
   for (uint32_t i = 0; i < num_pack_per_thread; i++)
   {
     if (thread_base_token + i * iter_stride < num_tokens)
     {
       *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
 #pragma unroll
       for (uint32_t j = 0; j < pack_size; j++)
       {
         x_val_float[i][j] = convert_to_float(x_val[i][j]);
       }
 
       if constexpr (sub_mean)
       {
 #pragma unroll
         for (uint32_t j = 0; j < pack_size; j++)
         {
           x_val_float[i][j] -= mean_val_float[j];
         }
       }
 
       if constexpr (has_sm_scale)
       {
 #pragma unroll
         for (uint32_t j = 0; j < pack_size; j++)
         {
           x_val_float[i][j] *= sm_scale;
         }
       }
     }
     else
     {
 #pragma unroll
       for (uint32_t j = 0; j < pack_size; j++)
       {
         x_val_float[i][j] = 0.0f;
       }
     }
   }
 
   float amax_val = 0.0000001f;
 
 #pragma unroll
   for (uint32_t i = 0; i < num_pack_per_thread; i++)
   {
 #pragma unroll
     for (uint32_t j = 0; j < pack_size; j++)
     {
       // 隔离极值：消除 padding 的 0 对量化极值 amax 的影响
       uint32_t current_dim = (thread_id % num_threads_per_token) * pack_size + j;
       if (current_dim < actual_head_dim) {
           amax_val = fmaxf(amax_val, fabsf(x_val_float[i][j]));
       } else {
           x_val_float[i][j] = 0.0f; 
       }
     }
   }
 
   __shared__ float s_amax;
   const float block_amax_val = vllm::blockReduceMax(amax_val);
   if (thread_id == 0)
   {
     s_amax = block_amax_val;
     scale_ptr_base[0] = s_amax / 127.0f;
   }
 
   __syncthreads();
 
   float tmp_scale = 127.0f / s_amax;
 
   constexpr uint32_t num_char4_per_pack = pack_size / 4;
   char4 o_val[num_pack_per_thread][num_char4_per_pack];
 
 #pragma unroll
   for (uint32_t i = 0; i < num_pack_per_thread; i++)
   {
 #pragma unroll
     for (uint32_t j = 0; j < num_char4_per_pack; j += 1)
     {
       o_val[i][j] = make_char4(
         float_to_int8_rn(x_val_float[i][j * 4 + 0] * tmp_scale),
         float_to_int8_rn(x_val_float[i][j * 4 + 1] * tmp_scale),
         float_to_int8_rn(x_val_float[i][j * 4 + 2] * tmp_scale),
         float_to_int8_rn(x_val_float[i][j * 4 + 3] * tmp_scale)
       );
     }
   }
 
 #pragma unroll
   for (uint32_t i = 0; i < num_pack_per_thread; i++)
   {
     if (thread_base_token + i * iter_stride < num_tokens)
     {
       if constexpr (pack_size == 8)
       {
         *reinterpret_cast<float2*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<float2*>(&o_val[i][0]);
       }
       else if constexpr (pack_size == 4)
       {
         *reinterpret_cast<uint32_t*>(output_ptr_base + i * iter_stride * stride_seq_output) = *reinterpret_cast<uint32_t*>(&o_val[i][0]);
       }
     }
   }
 }
 
 template <uint32_t head_dim, uint32_t BLOCK_SIZE, uint32_t num_pack_per_thread = 1, typename T>
 __global__ void SubMeanKernel(T *__restrict__ input, T *__restrict__ mean, T *__restrict__ output, const uint32_t num_tokens,
                             const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                             const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                             const uint32_t stride_bz_output, const uint32_t stride_seq_output, const uint32_t stride_h_output)
 {
   static_assert(num_pack_per_thread > 0, "The number of pack per thread must be greater than 0");
 
   using T2 = typename std::conditional<
     std::is_same<T, float>::value,
     float2,
     typename std::conditional<
       std::is_same<T, half>::value,
       half2,
       nv_bfloat162
     >::type
   >::type;
 
   using OutputT2 = typename std::conditional<
     std::is_same<T, float>::value,
     float2,
     half2
   >::type;
 
   constexpr uint32_t pack_size = PackTraits<T>::pack_size;
   constexpr uint32_t num_threads_per_token = head_dim / pack_size;
 
   static_assert(num_threads_per_token <= 64, "The number of threads per token must be less than or equal to warp size");
 
   T x_val[num_pack_per_thread][pack_size];
   T mean_val[pack_size];
   T2 x_val_conv[num_pack_per_thread][pack_size / 2];
   T2 mean_val_conv[pack_size / 2];
 
   uint32_t bx = blockIdx.x;
   uint32_t head_id = blockIdx.y;
   uint32_t batch_id = blockIdx.z;
   uint32_t thread_id = threadIdx.x;
 
   uint32_t thread_base_token = bx * BLOCK_SIZE + thread_id / num_threads_per_token;
   T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
   T *mean_ptr_base = mean + batch_id * stride_bz_mean + head_id * stride_h_mean + thread_id % num_threads_per_token * pack_size;
   T *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + thread_base_token * stride_seq_output + thread_id % num_threads_per_token * pack_size;
 
   *(float4*)(&mean_val[0]) = *(float4*)(mean_ptr_base);
 
 #pragma unroll
   for (uint32_t j = 0; j < pack_size / 2; j++)
   {
     if constexpr (std::is_same<T, float>::value)
     {
       mean_val_conv[j] = make_float2(mean_val[j * 2], mean_val[j * 2 + 1]);
     }
     else if constexpr (std::is_same<T, half>::value)
     {
       mean_val_conv[j] = ((half2*)mean_val)[j];
     }
     else
     {
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 800
       mean_val_conv[j] = ((nv_bfloat162*)mean_val)[j];
 #endif
     }
   }
 
   constexpr uint32_t iter_stride = BLOCK_SIZE / num_pack_per_thread;
 
   for (uint32_t i = 0; i < num_pack_per_thread; i++)
   {
     if (thread_base_token + i * iter_stride < num_tokens)
     {
       *(float4*)(&x_val[i][0]) = *(float4*)(input_ptr_base + i * iter_stride * stride_seq_input);
 
 #pragma unroll
       for (uint32_t j = 0; j < pack_size / 2; j++)
       {
         if constexpr (std::is_same<T, float>::value)
         {
           x_val_conv[i][j] = make_float2(x_val[i][j * 2], x_val[i][j * 2 + 1]);
           x_val_conv[i][j] = make_float2(
             x_val_conv[i][j].x - mean_val_conv[j].x,
             x_val_conv[i][j].y - mean_val_conv[j].y
           );
         }
         else if constexpr (std::is_same<T, half>::value)
         {
           x_val_conv[i][j] = ((half2*)x_val[i])[j];
           x_val_conv[i][j] = __hsub2(x_val_conv[i][j], mean_val_conv[j]);
         }
         else
         {
 #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 800
           x_val_conv[i][j] = ((nv_bfloat162*)x_val[i])[j];
           x_val_conv[i][j] = __hsub2(x_val_conv[i][j], mean_val_conv[j]);
 #endif
         }
       }
 
       if constexpr (std::is_same<T, float>::value)
       {
         *(float4*)(&output_ptr_base[i * iter_stride * stride_seq_output]) = *(float4*)(&x_val_conv[i][0]);
       }
       else
       {
         *(float4*)(output_ptr_base + i * iter_stride * stride_seq_output) = *(float4*)(&x_val_conv[i][0]);
       }
     }
   }
 }
 
 template <uint32_t head_dim, uint32_t CTA_SIZE, bool pad_zero=false, typename T>
 __global__ void TransposePadPermuteKernel(T *__restrict__ input, T *__restrict__ output, const uint32_t num_tokens,
                             const uint32_t stride_bz_input, const uint32_t stride_seq_input, const uint32_t stride_h_input,
                             const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output)
 {
   static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "TransposePadPermuteKernel only supports half-precision types due to shared memory constraints");
 
   constexpr uint32_t pack_size = 8;
   uint32_t num_threads_per_token = head_dim / pack_size;
   uint32_t num_threads_per_cta = CTA_SIZE / pack_size;
 
   uint32_t bx = blockIdx.x;
   uint32_t head_id = blockIdx.y;
   uint32_t batch_id = blockIdx.z;
   uint32_t thread_id = threadIdx.x;
 
   uint32_t thread_base_token = bx * CTA_SIZE + thread_id / num_threads_per_token;
 
   T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + thread_base_token * stride_seq_input + thread_id % num_threads_per_token * pack_size;
   T* output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + bx * CTA_SIZE + thread_id % num_threads_per_cta * pack_size + thread_id / num_threads_per_cta * stride_d_output;
 
   __shared__ T shared_load[CTA_SIZE][head_dim];
   __shared__ T shared_store[head_dim][CTA_SIZE];
 
   uint32_t smem_load_row_base = ((thread_id / num_threads_per_token) / 16) * 16;
   uint32_t smem_load_row_mod = (thread_id / num_threads_per_token) % 16;
   uint32_t smem_load_row = smem_load_row_base + (smem_load_row_mod  / 8) * 2 + ((smem_load_row_mod / 2) % 4) * 4 + (smem_load_row_mod % 2);
 
   constexpr cp_async::SharedMemFillMode fill_mode = pad_zero ? cp_async::SharedMemFillMode::kFillZero : cp_async::SharedMemFillMode::kNoFill;
   cp_async::pred_load_128b<cp_async::PrefetchMode::kNoPrefetch, fill_mode>(shared_load[smem_load_row] + thread_id % num_threads_per_token * pack_size, input_ptr_base, thread_base_token < num_tokens);
   cp_async::commit_group();
   cp_async::wait_group<0>();
   __syncthreads();
 
   uint32_t smem_row_base = thread_id % CTA_SIZE;
   uint32_t smem_col_base = thread_id / CTA_SIZE;
   uint32_t smem_col_stride = head_dim / 8;
 
 #pragma unroll
   for (uint32_t i = 0; i < 8; i++)
   {
     shared_store[smem_col_base + i * smem_col_stride][smem_row_base] = shared_load[smem_row_base][smem_col_base + i * smem_col_stride];
   }
 
   __syncthreads();
 
   *(float4*)(output_ptr_base) = *(float4*)(&shared_store[thread_id / num_threads_per_cta][thread_id % num_threads_per_cta * pack_size]);
 }
 
 template<uint32_t pad_size, bool sub_mean = false, typename T>
 __global__ void MeanScaleKernel(T *__restrict__ input, int8_t *__restrict__ output, float *__restrict__ mean, float *__restrict__ scale, const float scale_max, const uint32_t num_tokens,
                             const uint32_t stride_bz_input, const uint32_t stride_d_input, const uint32_t stride_h_input,
                             const uint32_t stride_bz_output, const uint32_t stride_d_output, const uint32_t stride_h_output,
                             const uint32_t stride_bz_mean, const uint32_t stride_h_mean,
                             const uint32_t stride_bz_scale, const uint32_t stride_h_scale)
 {
   static_assert(std::is_same<T, half>::value || std::is_same<T, nv_bfloat16>::value, "MeanScaleKernel only supports half-precision types");
 
   constexpr uint32_t pack_size = 8;
 
   uint32_t head_id = blockIdx.x;
   uint32_t batch_id = blockIdx.y;
   uint32_t d_id = blockIdx.z;
   uint32_t thread_id = threadIdx.x;
 
   uint32_t num_threads = blockDim.x;
   uint32_t gmem_stride = num_threads * pack_size;
   uint32_t fp8_padded_num_tokens = (num_tokens + 15) / 16 * 16;
   uint32_t num_iters = fp8_padded_num_tokens / gmem_stride + ((fp8_padded_num_tokens % gmem_stride) > thread_id * pack_size);
 
   T *input_ptr_base = input + batch_id * stride_bz_input + head_id * stride_h_input + d_id * stride_d_input + thread_id * pack_size;
   int8_t *output_ptr_base = output + batch_id * stride_bz_output + head_id * stride_h_output + d_id * stride_d_output + thread_id * pack_size;
 
   T x_val[8];
   float x_val_float[8];
   uint32_t x_val_fp8[2];
 
   float max_val = - 1000000.0f;
   float min_val = 1000000.0f;
   float sum_val = 0.0f;
 
   for (int i = 0; i < num_iters; i++)
   {
     *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
 #pragma unroll
     for (uint32_t j = 0; j < 8; j++)
     {
       float x_temp = convert_to_float(x_val[j]);
       max_val = fmaxf(max_val, x_temp);
       min_val = fminf(min_val, x_temp);
 
       if constexpr (sub_mean)
       {
         sum_val += x_temp;
       }
     }
   }
 
   __shared__ float s_amax_val;
   __shared__ float s_mean_val;
 
   float block_max_val = vllm::blockReduceMax(max_val);
   float block_min_val = vllm::blockReduceMin(min_val);
   float block_sum_val;
 
   if constexpr (sub_mean)
   {
     block_sum_val = vllm::blockReduceSum(sum_val);
   }
 
   if (thread_id == 0)
   {
     s_mean_val = block_sum_val / fp8_padded_num_tokens;
 
     if constexpr (sub_mean)
     {
       s_amax_val = fmaxf(fabsf(block_max_val - s_mean_val), fabsf(block_min_val - s_mean_val));
       mean[batch_id * stride_bz_mean + head_id * stride_h_mean + d_id] = s_mean_val;
     }
     else
     {
       s_amax_val = fmaxf(fabsf(block_max_val), fabsf(block_min_val));
     }
 
     scale[batch_id * stride_bz_scale + head_id * stride_h_scale + d_id] = s_amax_val / scale_max;
   }
 
   __syncthreads();
 
   float mean_val = s_mean_val;
   float recp_scale = scale_max / s_amax_val;
 
   uint32_t padded_num_tokens = (num_tokens + pad_size - 1) / pad_size * pad_size;
   num_iters = padded_num_tokens / gmem_stride + ((padded_num_tokens % gmem_stride) > thread_id * pack_size);
 
   for (int i = 0; i < num_iters; i++)
   {
     *(float4*)(&x_val[0]) = *(float4*)(input_ptr_base + i * gmem_stride);
 #pragma unroll
     for (uint32_t j = 0; j < 8; j++)
     {
       x_val_float[j] = convert_to_float(x_val[j]);
       if constexpr (sub_mean)
       {
         x_val_float[j] = (x_val_float[j] - mean_val) * recp_scale;
       }
       else
       {
         x_val_float[j] *= recp_scale;
       }
     }
 
     floatx4_to_e4m3x4(x_val_fp8, x_val_float, x_val_float + 2);
     floatx4_to_e4m3x4(x_val_fp8 + 1, x_val_float + 4, x_val_float + 6);
 
     *(uint2*)(output_ptr_base + i * gmem_stride) = *(uint2*)(&x_val_fp8[0]);
   }
 }
 
 void quant_per_block_int8_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 torch::Tensor scale,
                 float sm_scale,
                 int block_size,
                 int tensor_layout,
                 int actual_head_dim)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
   CHECK_CUDA(scale);
   
   CHECK_DTYPE(output, torch::kInt8);
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_seq_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_seq_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   auto input_dtype = input.scalar_type();
   uint32_t a_hd = actual_head_dim > 0 ? actual_head_dim : head_dim;
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
     DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
       DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
 
         CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
         CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
         constexpr int pack_size = KernelLaunchConfig<c_type>::pack_size;
         constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / pack_size) + 1023) / 1024;
 
         dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
 
         dim3 block(BLOCK_SIZE * (HEAD_DIM / pack_size) / num_pack_per_thread);
 
         QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, true, false, c_type><<<grid, block>>>(
           reinterpret_cast<c_type*>(input.data_ptr()),
           nullptr,
           output.data_ptr<int8_t>(),
           reinterpret_cast<float*>(scale.data_ptr()),
           sm_scale,
           num_tokens,
           stride_bz_input, stride_seq_input, stride_h_input,
           0, 0,
           stride_bz_output, stride_seq_output, stride_h_output,
           scale.stride(0), scale.stride(1), a_hd
         );
       });
     });
   });
 }
 
 void quant_per_block_int8_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 torch::Tensor scale,
                 int block_size,
                 int tensor_layout,
                 int actual_head_dim)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
   CHECK_CUDA(scale);
   
   CHECK_DTYPE(output, torch::kInt8);
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_seq_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_seq_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   auto input_dtype = input.scalar_type();
   uint32_t a_hd = actual_head_dim > 0 ? actual_head_dim : head_dim;
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
     DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
       DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
 
         CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
         CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
         constexpr int pack_size = KernelLaunchConfig<c_type>::pack_size;
         constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / pack_size) + 1023) / 1024;
 
         dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
 
         dim3 block(BLOCK_SIZE * (HEAD_DIM / pack_size) / num_pack_per_thread);
 
         QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
           reinterpret_cast<c_type*>(input.data_ptr()),
           nullptr,
           output.data_ptr<int8_t>(),
           reinterpret_cast<float*>(scale.data_ptr()),
           0.0f,
           num_tokens,
           stride_bz_input, stride_seq_input, stride_h_input,
           0, 0,
           stride_bz_output, stride_seq_output, stride_h_output,
           scale.stride(0), scale.stride(1), a_hd
         );
       });
     });
   });
 }
 
 void quant_per_block_int8_fuse_sub_mean_cuda(
                 torch::Tensor input,
                 torch::Tensor mean,
                 torch::Tensor output,
                 torch::Tensor scale,
                 int block_size,
                 int tensor_layout,
                 int actual_head_dim)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(mean);
   CHECK_CUDA(output);
   CHECK_CUDA(scale);
   
   CHECK_DTYPE(output, torch::kInt8);
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(mean);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(mean, 3);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_seq_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_seq_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   auto input_dtype = input.scalar_type();
   auto mean_dtype = mean.scalar_type();
 
   TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");
   uint32_t a_hd = actual_head_dim > 0 ? actual_head_dim : head_dim;
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
     DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
       DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
 
         CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
         CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
         CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE);
 
         constexpr int pack_size = KernelLaunchConfig<c_type>::pack_size;
         constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / pack_size) + 1023) / 1024;
 
         dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
 
         dim3 block(BLOCK_SIZE * (HEAD_DIM / pack_size) / num_pack_per_thread);
 
         QuantInt8Kernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread, false, true, c_type><<<grid, block>>>(
           reinterpret_cast<c_type*>(input.data_ptr()),
           reinterpret_cast<c_type*>(mean.data_ptr()),
           output.data_ptr<int8_t>(),
           reinterpret_cast<float*>(scale.data_ptr()),
           0.0f,
           num_tokens,
           stride_bz_input, stride_seq_input, stride_h_input,
           mean.stride(0), mean.stride(1),
           stride_bz_output, stride_seq_output, stride_h_output,
           scale.stride(0), scale.stride(1), a_hd
         );
       });
     });
   });
 }
 
 void quant_per_warp_int8_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 torch::Tensor scale,
                 int block_size,
                 int warp_block_size,
                 int tensor_layout,
                 int actual_head_dim)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
   CHECK_CUDA(scale);
   
   CHECK_DTYPE(output, torch::kInt8);
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_seq_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_seq_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   auto input_dtype = input.scalar_type();
   // 使用 actual_head_dim 向下传递
   uint32_t a_hd = actual_head_dim > 0 ? actual_head_dim : head_dim;
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
     DISPATCH_BLOCK_SIZE(block_size, BLOCK_SIZE, {
       DISPATCH_WARP_BLOCK_SIZE(warp_block_size, WARP_BLOCK_SIZE, {
         DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
 
           CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
           CHECK_SHAPE(scale, batch_size, num_heads, (num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE));
 
           dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE * (BLOCK_SIZE / WARP_BLOCK_SIZE), num_heads, batch_size);
 
           constexpr int pack_size = KernelLaunchConfig<c_type>::pack_size;
           constexpr int num_pack_per_thread = (WARP_BLOCK_SIZE * (HEAD_DIM / pack_size) + 1023) / 1024;
 
           dim3 block(WARP_BLOCK_SIZE * (HEAD_DIM / pack_size) / num_pack_per_thread);
 
           QuantInt8Kernel<HEAD_DIM, WARP_BLOCK_SIZE, num_pack_per_thread, false, false, c_type><<<grid, block>>>(
             reinterpret_cast<c_type*>(input.data_ptr()),
             nullptr,
             output.data_ptr<int8_t>(),
             reinterpret_cast<float*>(scale.data_ptr()),
             0.0,
             num_tokens,
             stride_bz_input, stride_seq_input, stride_h_input,
             0, 0,
             stride_bz_output, stride_seq_output, stride_h_output,
             scale.stride(0), scale.stride(1), a_hd
           );
         });
       });
     });
   });
 }
 
 void sub_mean_cuda(
                 torch::Tensor input,
                 torch::Tensor mean,
                 torch::Tensor output,
                 int tensor_layout)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(mean);
   CHECK_CUDA(output);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(mean);
   CHECK_CONTIGUOUS(output);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(mean, 3);
   CHECK_DIMS(output, 4);
 
   auto input_dtype = input.scalar_type();
   auto output_dtype = output.scalar_type();
   TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type for native dtype support");
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_seq_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_seq_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_seq_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   auto mean_dtype = mean.scalar_type();
 
   TORCH_CHECK(input_dtype == mean_dtype, "Input and mean must have the same data type");
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input_dtype, c_type, {
     DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
 
         CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
         CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
 
         constexpr int BLOCK_SIZE = (HEAD_DIM == 128) ? 64 : 128;
         constexpr int pack_size = KernelLaunchConfig<c_type>::pack_size;
         constexpr int num_pack_per_thread = (BLOCK_SIZE * (HEAD_DIM / pack_size) + 1023) / 1024;
 
         dim3 grid((num_tokens + BLOCK_SIZE - 1) / BLOCK_SIZE, num_heads, batch_size);
 
         dim3 block(BLOCK_SIZE * (HEAD_DIM / pack_size) / num_pack_per_thread);
 
         SubMeanKernel<HEAD_DIM, BLOCK_SIZE, num_pack_per_thread><<<grid, block>>>(
           reinterpret_cast<c_type*>(input.data_ptr()),
           reinterpret_cast<c_type*>(mean.data_ptr()),
           reinterpret_cast<c_type*>(output.data_ptr()),
           num_tokens,
           stride_bz_input, stride_seq_input, stride_h_input,
           mean.stride(0), mean.stride(1),
           stride_bz_output, stride_seq_output, stride_h_output
         );
     });
   });
 }
 
 void transpose_pad_permute_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 int tensor_layout)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
 
   CHECK_LASTDIM_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
 
   constexpr int CTA_SIZE = 64;
 
   const int batch_size = input.size(0);
   const int head_dim = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_tokens, padded_num_tokens, num_heads;
   int stride_seq_input, stride_h_input, stride_d_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_tokens = input.size(1);
     num_heads = input.size(2);
     stride_seq_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_d_output = output.stride(1);
     stride_h_output = output.stride(2);
 
     padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
 
     CHECK_SHAPE(output, batch_size, head_dim, num_heads, padded_num_tokens);
   }
   else
   {
     num_tokens = input.size(2);
     num_heads = input.size(1);
     stride_seq_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_d_output = output.stride(2);
     stride_h_output = output.stride(1);
 
     padded_num_tokens = (num_tokens + CTA_SIZE - 1) / CTA_SIZE * CTA_SIZE;
     CHECK_SHAPE(output, batch_size, num_heads, head_dim, padded_num_tokens);
   }
 
   auto input_dtype = input.scalar_type();
   auto output_dtype = output.scalar_type();
 
   TORCH_CHECK(input_dtype == output_dtype, "Input and output must have the same data type");
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_NOFP32(input_dtype, c_type, {
    DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, {
      if constexpr (HEAD_DIM == 256) {
          TORCH_CHECK(false, "transpose_pad_permute_cuda is not supported for head_dim=256");
      } else {
          // 骗过编译器的关键：如果等于256，用128骗过静态检查，反正这段代码运行时绝不会被执行
          constexpr int SAFE_HEAD_DIM = (HEAD_DIM == 256) ? 128 : HEAD_DIM;
          
          dim3 grid(padded_num_tokens / CTA_SIZE, num_heads, batch_size);

          static_assert(CTA_SIZE * SAFE_HEAD_DIM <= 8192);

          dim3 block(CTA_SIZE * (SAFE_HEAD_DIM / 8));

          TransposePadPermuteKernel<SAFE_HEAD_DIM, CTA_SIZE, true, c_type><<<grid, block>>>(
            reinterpret_cast<c_type*>(input.data_ptr()),
            reinterpret_cast<c_type*>(output.data_ptr()),
            num_tokens,
            stride_bz_input, stride_seq_input, stride_h_input,
            stride_bz_output, stride_d_output, stride_h_output
          );
      }
    });
  });
 }
 
 void scale_fuse_quant_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 torch::Tensor scale,
                 int num_tokens,
                 float scale_max,
                 int tensor_layout)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
   CHECK_CUDA(scale);
 
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int num_tokens_padded = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_heads, head_dim;
   int stride_d_input, stride_h_input, stride_d_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_heads = input.size(2);
     head_dim = input.size(1);
     stride_d_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_d_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_heads = input.size(1);
     head_dim = input.size(2);
     stride_d_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_d_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
   CHECK_SHAPE(scale, batch_size, num_heads, head_dim);
 
   constexpr int CTA_SIZE = 256;
 
   dim3 grid(num_heads, batch_size, head_dim);
   dim3 block(CTA_SIZE);
 
   auto input_dtype = input.scalar_type();
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_NOFP32(input_dtype, c_type, {
     MeanScaleKernel<64, false, c_type><<<grid, block>>>(
       reinterpret_cast<c_type*>(input.data_ptr()),
       reinterpret_cast<int8_t*>(output.data_ptr()),
       nullptr,
       reinterpret_cast<float*>(scale.data_ptr()),
       scale_max,
       num_tokens,
       stride_bz_input, stride_d_input, stride_h_input,
       stride_bz_output, stride_d_output, stride_h_output,
       0, 0,
       scale.stride(0), scale.stride(1)
     );
   });
 }
 
 void mean_scale_fuse_quant_cuda(
                 torch::Tensor input,
                 torch::Tensor output,
                 torch::Tensor mean,
                 torch::Tensor scale,
                 int num_tokens,
                 float scale_max,
                 int tensor_layout)
 {
   CHECK_CUDA(input);
   CHECK_CUDA(output);
   CHECK_CUDA(mean);
   CHECK_CUDA(scale);
 
   CHECK_DTYPE(mean, torch::kFloat);
   CHECK_DTYPE(scale, torch::kFloat);
 
   CHECK_CONTIGUOUS(input);
   CHECK_CONTIGUOUS(output);
   CHECK_CONTIGUOUS(mean);
   CHECK_CONTIGUOUS(scale);
 
   CHECK_DIMS(input, 4);
   CHECK_DIMS(output, 4);
   CHECK_DIMS(mean, 3);
   CHECK_DIMS(scale, 3);
 
   const int batch_size = input.size(0);
   const int num_tokens_padded = input.size(3);
 
   int stride_bz_input = input.stride(0);
   int stride_bz_output = output.stride(0);
 
   int num_heads, head_dim;
   int stride_d_input, stride_h_input, stride_d_output, stride_h_output;
 
   if (tensor_layout == 0)
   {
     num_heads = input.size(2);
     head_dim = input.size(1);
     stride_d_input = input.stride(1);
     stride_h_input = input.stride(2);
     stride_d_output = output.stride(1);
     stride_h_output = output.stride(2);
   }
   else
   {
     num_heads = input.size(1);
     head_dim = input.size(2);
     stride_d_input = input.stride(2);
     stride_h_input = input.stride(1);
     stride_d_output = output.stride(2);
     stride_h_output = output.stride(1);
   }
 
   CHECK_SHAPE(output, input.size(0), input.size(1), input.size(2), input.size(3));
   CHECK_SHAPE(mean, batch_size, num_heads, head_dim);
   CHECK_SHAPE(scale, batch_size, num_heads, head_dim);
 
   constexpr int CTA_SIZE = 256;
 
   dim3 grid(num_heads, batch_size, head_dim);
   dim3 block(CTA_SIZE);
 
   auto input_dtype = input.scalar_type();
 
   DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16_NOFP32(input_dtype, c_type, {
     MeanScaleKernel<64, true, c_type><<<grid, block>>>(
       reinterpret_cast<c_type*>(input.data_ptr()),
       reinterpret_cast<int8_t*>(output.data_ptr()),
       reinterpret_cast<float*>(mean.data_ptr()),
       reinterpret_cast<float*>(scale.data_ptr()),
       scale_max,
       num_tokens,
       stride_bz_input, stride_d_input, stride_h_input,
       stride_bz_output, stride_d_output, stride_h_output,
       mean.stride(0), mean.stride(1),
       scale.stride(0), scale.stride(1)
     );
   });
 }