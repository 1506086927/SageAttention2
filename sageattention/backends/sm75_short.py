"""
SM75 Short Attention Backend - v2 stride-aware kernel with fallback
"""
import torch
import torch.nn.functional as F
from .. import _sm75_short_sdpa_v2, _sm75_fast_dispatch

def sm75_short_sdpa(q, k, v, is_causal, sm_scale, layout="HND", return_lse=False):
    """
    三级 fallback 顺序：
    1. v2 stride-aware kernel (sm75_short_sdpa_v2)
    2. v1 fixed-layout kernel (sm75_custom_short_sdpa via _sm75_fast_dispatch.sm75_fast_sdpa)
    3. PyTorch SDPA
    """
    # Try v2 stride-aware kernel first (v2 doesn't support return_lse)
    if not return_lse:
        try:
            result_v2 = _sm75_short_sdpa_v2.sm75_short_sdpa_v2(q, k, v, is_causal, sm_scale if sm_scale is not None else 0.0, layout)
            if result_v2 is not None and result_v2.numel() > 0:
                return result_v2
        except Exception:
            pass
    
    # Fallback to v1 fixed-layout kernel
    try:
        is_nhd = (layout == "NHD")
        if is_nhd:
            q_tmp = q.transpose(1, 2).contiguous()
            k_tmp = k.transpose(1, 2).contiguous()
            v_tmp = v.transpose(1, 2).contiguous()
        else:
            q_tmp = q
            k_tmp = k
            v_tmp = v

        result_v1 = _sm75_fast_dispatch.sm75_fast_sdpa(
            q_tmp, k_tmp, v_tmp, is_causal, sm_scale if sm_scale is not None else 0.0, return_lse
        )
        if result_v1 is not None and result_v1.numel() > 0:
            if is_nhd:
                return result_v1.transpose(1, 2).contiguous()
            return result_v1
    except Exception:
        pass
    
    # Final fallback to PyTorch SDPA
    is_nhd = (layout == "NHD")
    q_pt = q.transpose(1, 2) if is_nhd else q
    k_pt = k.transpose(1, 2) if is_nhd else k
    v_pt = v.transpose(1, 2) if is_nhd else v

    if q_pt.size(1) != k_pt.size(1):
        num_groups = q_pt.size(1) // k_pt.size(1)
        k_pt = k_pt.repeat_interleave(num_groups, dim=1)
        v_pt = v_pt.repeat_interleave(num_groups, dim=1)

    out = F.scaled_dot_product_attention(
        q_pt, k_pt, v_pt, 
        is_causal=is_causal, 
        scale=sm_scale
    )
    if is_nhd:
        out = out.transpose(1, 2)
    return out