"""
AttentionSpec: 注意力计算的规格描述
"""
from dataclasses import dataclass
import torch


def _next_pad_size(x: int) -> int:
    """Pad head_dim to next supported size: 64, 128, or 256."""
    if x < 64:
        return 64
    elif x < 128:
        return 128
    elif x <= 256:
        return 256
    else:
        raise ValueError(f"Unsupported head_dim: {x}")


@dataclass(frozen=True)
class AttentionSpec:
    arch: str
    q_len: int
    kv_len: int
    head_dim: int
    padded_head_dim: int
    q_heads: int
    kv_heads: int
    layout: str          # "HND" or "NHD"
    dtype: torch.dtype
    is_causal: bool
    return_lse: bool

    @classmethod
    def from_tensors(cls, q, k, v, layout, is_causal, return_lse, arch):
        seq_dim = 2 if layout == "HND" else 1
        head_dim_dim = 3  # last dimension is head_dim for both HND and NHD
        q_len = q.size(seq_dim)
        kv_len = k.size(seq_dim)
        head_dim = q.size(head_dim_dim)
        padded_head_dim = _next_pad_size(head_dim)
        q_heads = q.size(1) if layout == "HND" else q.size(2)
        kv_heads = k.size(1) if layout == "HND" else k.size(2)
        return cls(arch, q_len, kv_len, head_dim, padded_head_dim,
                    q_heads, kv_heads, layout, q.dtype, is_causal, return_lse)
