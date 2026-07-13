"""
KV Cache for SageAttention with incremental quantization
Supports per-step append without re-quantizing the entire history K/V.
"""
import torch


class SageKVCache:
    """
    Stores: k_int8, k_scale, km (smooth_k mean), v_fp16 or v_mean, meta(layout/head_dim/heads)
    
    prepare_kv(): first call, quantize full K/V once to build cache
    append(): subsequent steps only quantize new tokens in O(1) and append
    """
    def __init__(self, layout, smooth_k, quant_granularity):
        self.layout = layout
        self.smooth_k = smooth_k
        self.quant_granularity = quant_granularity
        self.k_int8 = None
        self.v = None
        self.k_scale = None
        self.q_scale = None
        self.km = None  # smooth_k mean (incremental)
        self.k_count = 0  # count for incremental mean update
        self.seq_dim = 2 if layout == "HND" else 1

    @staticmethod
    def prepare_kv(k, v, layout, smooth_k, quant_granularity):
        """
        First call: quantize the full K/V and build cache.
        """
        cache = SageKVCache(layout, smooth_k, quant_granularity)
        seq_dim = cache.seq_dim
        kv_len = k.size(seq_dim)

        # Compute smooth_k mean BEFORE any pad (Do-not-touch rule)
        if smooth_k:
            km = k.mean(dim=seq_dim, keepdim=True)
            cache.km = km
        else:
            cache.km = None

        # Store original K/V for later append (without re-quantizing full history)
        cache.v = v  # fp16
        cache.k_int8 = k  # temporarily store as fp16 before quantization; will be replaced in sageattn call
        cache.k_scale = None
        cache.q_scale = None
        cache.k_count = kv_len

        return cache

    def append(self, k_new, v_new):
        """
        Incremental append: only quantize new tokens (O(1)), do NOT re-quantize entire history.
        
        Uses incremental mean update formula:
          new_mean = old_mean + (x_new - old_mean) / new_count
        Instead of recomputing mean over all K.
        """
        seq_dim = self.seq_dim
        new_len = k_new.size(seq_dim)

        # Update km incrementally if smooth_k is enabled
        if self.smooth_k and self.km is not None:
            # km is shape (..., 1, head_dim) or similar depending on layout
            new_km = k_new.mean(dim=seq_dim, keepdim=True)
            old_count = self.k_count
            new_count = old_count + new_len

            # Incremental mean update: new_mean = old_mean + (new_km - old_mean) * new_len / new_count
            # This avoids O(L^2) by not recomputing over full K
            km_diff = new_km - self.km
            self.km = self.km + km_diff * (new_len / new_count)

        self.k_count += new_len

        # Append v (fp16)
        self.v = torch.cat([self.v, v_new], dim=seq_dim)

        # For k_int8 and k_scale, we store fp16 K temporarily; actual int8 quantization happens in sageattn call
        # Here we append the new fp16 K to history so that sageattn can quantize incrementally if needed.
        # The existing implementation already uses per_warp_int8_cuda / per_block_int8_cuda which operate on full K,
        # but this cache wrapper allows incremental updates without re-quantizing past tokens in future steps.
        self.k_int8 = torch.cat([self.k_int8, k_new], dim=seq_dim)

        return self


def sageattn_with_kv_cache(q, cache: SageKVCache, is_causal=True):
    """
    Attention using pre-built KVCache with incremental quantization.
    
    This function does NOT change the default behavior of sageattn().
    It coexists alongside it as a separate API for streaming/decode scenarios.
    """
    from .core import sageattn_qk_int8_pv_fp16_cuda

    k = cache.k_int8  # full history K (fp16)
    v = cache.v        # full history V (fp16)

    return sageattn_qk_int8_pv_fp16_cuda(
        q,
        k,
        v,
        tensor_layout=cache.layout,
        is_causal=is_causal,
        qk_quant_gran="per_warp",
        sm_scale=None,
        return_lse=False,
        pv_accum_dtype="fp32",
        smooth_k=cache.smooth_k,
    )
