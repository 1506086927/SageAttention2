"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn.functional as F

from .triton.quant_per_block import per_block_int8 as per_block_int8_triton
from .triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
from .triton.attn_qk_int8_per_block import forward as attn_false
from .triton.attn_qk_int8_per_block_causal import forward as attn_true
from .triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
from .triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from .triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton

try:
    from . import _qattn_sm80
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from . import _qattn_sm89
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    from . import _qattn_sm90
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

from .quant import per_block_int8 as per_block_int8_cuda
from .quant import per_warp_int8 as per_warp_int8_cuda
from .quant import sub_mean
from .quant import per_channel_fp8

from typing import Any, List, Literal, Optional, Tuple, Union
import os
import warnings

# --- Performance & Robustness: Device capability cache ---
_device_sm_cache = {}

def _get_device_sm(device):
    """Safely get device capability per device context (Multi-GPU safe)."""
    dev_idx = device.index if device.index is not None else 0
    if dev_idx not in _device_sm_cache:
        major, minor = torch.cuda.get_device_capability(dev_idx)
        _device_sm_cache[dev_idx] = (major, minor)
    return _device_sm_cache[dev_idx]

_last_device_index = -1

def _set_device_if_needed(device):
    """Set CUDA device context only if it differs from current context."""
    global _last_device_index
    dev_index = device.index if device.index is not None else 0
    if _last_device_index != dev_index:
        torch.cuda.set_device(device)
        _last_device_index = dev_index

_SAGEATTEN_DEBUG = os.environ.get("SAGEATTEN_DEBUG", "0") == "1"

def _validate_inputs(q, k, v):
    """Validate inputs only in debug mode to avoid per-call overhead."""
    if _SAGEATTEN_DEBUG:
        assert q.is_cuda, "Input tensors must be on cuda."
        assert q.dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
        assert q.device == k.device == v.device, "All tensors must be on the same device."
        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
        assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    **kwargs: Any,
):
    if _SAGEATTEN_DEBUG:
        _validate_inputs(q, k, v)

    _set_device_if_needed(v.device)

    # 统一确保物理内存连续，解决 C++ 量化核基于 float4 128-bit 向量化加载越界的问题
    # 不强制篡改张量形状，保留外界期待的 tensor_layout 返回值
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    major, minor = _get_device_sm(q.device)
    arch = f"sm{major}{minor}"
    is_sm75 = (major == 7 and minor == 5)

    if arch == "sm90" and return_lse:
        raise NotImplementedError("return_lse is not supported on SM90.")

    # Prevent triple type conversion on SM75 (BF16/FP32 -> FP16 directly)
    orig_dtype = q.dtype
    if is_sm75 and q.dtype in [torch.bfloat16, torch.float32]:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    seq_len = q.size(2) if tensor_layout == "HND" else q.size(1)

    if arch == "sm86" or is_sm75:
        if is_sm75:
            # 对于长序列，回退到 CUDA 量化核心（原生支持 stride NHD）
            if seq_len >= 1024 or return_lse:
                result = sageattn_qk_int8_pv_fp16_cuda(
                    q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, 
                    qk_quant_gran="per_warp", sm_scale=sm_scale, 
                    return_lse=return_lse, pv_accum_dtype="fp32"
                )
                if return_lse:
                    o, lse = result
                    return o.to(orig_dtype), lse
                return result.to(orig_dtype)
            else:
                from . import _sm75_fast_dispatch
                is_nhd = (tensor_layout == "NHD")
                
                # 隔离内存操作：仅在送入假定 HND 物理布局的 C++ 核心前转置并强制连续
                if is_nhd:
                    q_tmp = q.transpose(1, 2).contiguous()
                    k_tmp = k.transpose(1, 2).contiguous()
                    v_tmp = v.transpose(1, 2).contiguous()
                else:
                    q_tmp = q
                    k_tmp = k
                    v_tmp = v

                result = _sm75_fast_dispatch.sm75_fast_sdpa(
                    q_tmp, k_tmp, v_tmp, is_causal, sm_scale if sm_scale is not None else 0.0
                )

                # 将结果转置回期望的 NHD 格式
                if is_nhd:
                    result = result.transpose(1, 2).contiguous()
                
                return result.to(orig_dtype)
        else:
            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm80":
        return sageattn_qk_int8_pv_fp16_triton(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse)
    elif arch == "sm89":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    elif arch == "sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32")
    elif arch == "sm120":
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout=tensor_layout, is_causal=is_causal, qk_quant_gran="per_warp", sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_triton(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    tensor_layout: str = "HND",
    quantization_backend: str = "cuda",
    is_causal: bool =False, 
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    original_dtype = q.dtype
    dtype = q.dtype
    _validate_inputs(q, k, v)
    _set_device_if_needed(v.device)

    major, minor = _get_device_sm(q.device)
    is_sm75 = (major == 7 and minor == 5)

    if q.dtype == torch.bfloat16 and is_sm75:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        dtype = torch.float16

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        q = torch.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 256 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout, actual_head_dim=head_dim_og)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    if is_causal:
        o, lse = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse, is_sm75=is_sm75)
    else:
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse, is_sm75=is_sm75)

    o = o[..., :head_dim_og]
    
    if o.dtype != original_dtype:
        o = o.to(original_dtype)

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_varlen(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    is_causal: bool = False,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    torch.cuda.set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        q = torch.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 256 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    major, minor = _get_device_sm(q.device)
    is_sm75 = (major == 7 and minor == 5)

    compute_dtype = dtype
    if dtype == torch.bfloat16 or dtype == torch.float32:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
        compute_dtype = torch.float16

    if smooth_k:
        km = k.mean(dim=0, keepdim=True)
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    from .triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
    from .triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen
    from .triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
    
    blkq_val = 64 if is_sm75 else 128
    
    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen_triton(
        q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, BLKQ=blkq_val, BLKK=64, sm_scale=sm_scale
    )
    
    if is_causal:
        o = attn_true_varlen(
            q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, 
            q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, 
            output_dtype=compute_dtype, is_sm75=is_sm75
        )
    else:
        o = attn_false_varlen(
            q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, 
            q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, 
            output_dtype=compute_dtype, is_sm75=is_sm75
        )

    o = o[..., :head_dim_og]
    
    if o.dtype != dtype:
        o = o.to(dtype)

    return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert SM80_ENABLED, "SM80 kernel is not available. Make sure your GPU compute capability is 8.0 or higher."
    _validate_inputs(q, k, v)
    assert qk_quant_gran in ["per_warp", "per_thread", "per_block"], "qk_quant_gran must be either 'per_warp', 'per_thread', or 'per_block'."

    _set_device_if_needed(v.device)

    major, minor = _get_device_sm(q.device)
    is_sm75 = (major == 7 and minor == 5)

    # Route out unsupported configuration for SM75
    if is_sm75 and qk_quant_gran == "per_block":
        qk_quant_gran = "per_warp"

    if is_sm75 and q.dtype in [torch.bfloat16, torch.float32]:
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2 # Default mapped
    if qk_quant_gran == "per_block":
        _qk_quant_gran = 2 
        qk_quant_gran = "per_warp" 

    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        q = torch.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 256 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    # PRE-CALCULATE smooth_k mean BEFORE padding avoiding zero dilution
    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        lse_correction_needed = return_lse
    else:
        km = None
        lse_correction_needed = False

    original_q_seq_len = q.size(seq_dim) if is_sm75 else None

    # Padding logic for SM75
    if is_sm75:
        q_len = q.size(seq_dim)
        kv_len = k.size(seq_dim)
        q_seq_pad = (64 - (q_len % 64)) % 64
        k_seq_pad = (32 - (kv_len % 32)) % 32

        if q_seq_pad > 0 or k_seq_pad > 0:
            if _tensor_layout == 0: 
                if q_seq_pad > 0:
                    q = torch.nn.functional.pad(q, (0, 0, 0, 0, 0, q_seq_pad))
                if k_seq_pad > 0:
                    k = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, k_seq_pad))
                    v = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, k_seq_pad))
            else: 
                if q_seq_pad > 0:
                    q = torch.nn.functional.pad(q, (0, 0, 0, q_seq_pad, 0, 0))
                if k_seq_pad > 0:
                    k = torch.nn.functional.pad(k, (0, 0, 0, k_seq_pad, 0, 0))
                    v = torch.nn.functional.pad(v, (0, 0, 0, k_seq_pad, 0, 0))

    lse_correction = None
    if lse_correction_needed and smooth_k and km is not None:
        try:
            h_qo = q.size(1) if _tensor_layout == 1 else q.size(2)
            h_kv = km.size(1) if _tensor_layout == 1 else km.size(2)
            num_kv_groups = h_qo // h_kv if h_kv > 0 else 1
            
            if num_kv_groups > 1:
                km_for_correction = km.repeat_interleave(num_kv_groups, dim=1) if _tensor_layout == 1 else km.repeat_interleave(num_kv_groups, dim=2)
            else:
                km_for_correction = km
            
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km_for_correction.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km_for_correction.transpose(2, 3)).squeeze(-1).to(torch.float32)
        except Exception as e:
            warnings.warn(f"Skipping lse_correction due to: {e}")
            lse_correction = None

    if is_sm75:
        BLKQ = 64  
        BLKK = 32  
        WARPQ = 16  
        WARPK = 32  
    else:
        BLKQ = 128
        BLKK = 64
        WARPQ = 16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32
        WARPK = 64

    actual_head_dim = head_dim_og

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(
            q, k, km, tensor_layout=tensor_layout,
            BLKQ=BLKQ, WARPQ=WARPQ, BLKK=BLKK, actual_head_dim=actual_head_dim
        )
    elif qk_quant_gran == "per_block":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(
            q, k, km, tensor_layout=tensor_layout,
            BLKQ=BLKQ, BLKK=BLKK, sm_scale=sm_scale, actual_head_dim=actual_head_dim
        )
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(
            q, k, km, tensor_layout=tensor_layout,
            BLKQ=BLKQ, WARPQ=WARPQ, BLKK=BLKK, WARPK=WARPK
        )

    o = torch.empty(q.size(), dtype=torch.float16, device=q.device)

    _pv_accum_dtype = pv_accum_dtype
    if _pv_accum_dtype == "fp32+fp16":
        _pv_accum_dtype = "fp16+fp32"

    if _pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {_pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if _pv_accum_dtype == 'fp32':
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif _pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif _pv_accum_dtype == "fp16+fp32":
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if is_sm75 and original_q_seq_len is not None:
        if _tensor_layout == 0:
            o = o[:, :original_q_seq_len, :, :]
        else:
            o = o[:, :, :original_q_seq_len, :]

    if o.dtype != dtype:
        o = o.to(dtype)

    if return_lse:
        if is_sm75 and original_q_seq_len is not None:
            lse = lse[:, :, :original_q_seq_len]
            if lse_correction is not None:
                lse_correction = lse_correction[:, :, :original_q_seq_len]
        if lse_correction is not None:
            return o, lse / 1.44269504 + lse_correction * sm_scale
        else:
            return o, lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    _validate_inputs(q, k, v)
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."

    _set_device_if_needed(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        q = torch.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 256 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    actual_head_dim = head_dim_og

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, actual_head_dim=actual_head_dim)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    v_fp8, v_scale, vm = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=smooth_v)

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    _validate_inputs(q, k, v)
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."

    _set_device_if_needed(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        q = torch.nn.functional.pad(q, (0, 256 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 256 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = k.mean(dim=seq_dim, keepdim=True)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    actual_head_dim = head_dim_og

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, actual_head_dim=actual_head_dim)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)

    o = torch.empty(q.size(), dtype=dtype, device=q.device)

    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
        else:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_fp8, v_scale, _ = per_channel_fp8(v, tensor_layout=tensor_layout, smooth_v=False)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o