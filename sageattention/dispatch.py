"""
Dispatch logic for attention backends
"""
from functools import lru_cache
from .specs import AttentionSpec


@lru_cache(maxsize=256)
def select_backend(spec: AttentionSpec):
    """
    纯函数：给定 spec（不含实际 tensor 数据，只含形状/dtype/flag），返回一个 backend 名字或 callable。
    lru_cache 生效的前提是 AttentionSpec 是 frozen dataclass（可 hash），
    绝不能把 tensor 本身放进 spec 里，否则 lru_cache 会持有 tensor 引用导致显存泄漏。
    """
    if spec.arch != "sm75":
        return "torch_sdpa_or_other_backend"
    
    route_len = max(spec.q_len, spec.kv_len)
    
    if route_len < 1024 and not spec.return_lse:
        return "sm75_short"
    
    return "sm75_qattn_long"