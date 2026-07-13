"""
Utility functions for tensor shape manipulation
"""
import torch


def pad_head_dim(x: torch.Tensor, head_dim_og: int) -> torch.Tensor:
    """Pad head_dim to next supported size: 64, 128, or 256."""
    if head_dim_og < 64:
        return torch.nn.functional.pad(x, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        return torch.nn.functional.pad(x, (0, 128 - head_dim_og))
    elif head_dim_og > 128 and head_dim_og <= 256:
        return torch.nn.functional.pad(x, (0, 256 - head_dim_og))
    elif head_dim_og > 256:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")
    return x


def ensure_lastdim_contiguous(x: torch.Tensor) -> torch.Tensor:
    """Ensure the last dimension is contiguous, only transpose if necessary."""
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


def compute_lse_correction(q: torch.Tensor, km: torch.Tensor, tensor_layout: str) -> torch.Tensor:
    """Calculate LSE correction when smooth_k is applied."""
    h_qo = q.size(1) if tensor_layout == "HND" else q.size(2)
    h_kv = km.size(1) if tensor_layout == "HND" else km.size(2)
    
    num_kv_groups = h_qo // h_kv if h_kv > 0 else 1
    
    if num_kv_groups > 1:
        km_for_correction = km.repeat_interleave(num_kv_groups, dim=1 if tensor_layout == "HND" else 2)
    else:
        km_for_correction = km
    
    if tensor_layout == "NHD":
        return torch.matmul(q.transpose(1, 2), km_for_correction.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        return torch.matmul(q, km_for_correction.transpose(2, 3)).squeeze(-1).to(torch.float32)