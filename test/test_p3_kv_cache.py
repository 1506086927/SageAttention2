"""
测试脚本：SageAttention P3 验证 - KV Cache 增量量化
模拟 VoxCPM 逐 token decode 场景，验证:
1) sageattn_with_kv_cache 每步耗时随 kv_len 线性增长（而非二次增长）
2) 与 sageattn() 全量重算的结果数值一致（allclose）
"""
import torch
import torch.nn.functional as F
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sageattention import sageattn
from sageattention.kv_cache import SageKVCache, sageattn_with_kv_cache


def allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2):
    if out_new is None or out_ref is None:
        return False, "One of the outputs is None"
    try:
        return torch.allclose(out_new, out_ref, atol=atol, rtol=rtol), None
    except Exception as e:
        return False, str(e)


def test_p3_linear_growth():
    """
    Test P3-2: Simulate streaming decode (kv_len from 1 to 256).
    Verify that sageattn_with_kv_cache latency grows linearly, not quadratically.
    """
    print("\n=== Testing P3-2: Linear growth of kv_len ===")

    # Use a fixed Q shape and growing KV length
    batch = 1
    q_heads = 8
    kv_heads = 8
    head_dim = 64
    layout = "HND"

    # Pre-generate K/V tokens (for streaming simulation)
    max_kv_len = 256
    k_all = torch.randn(batch, kv_heads, max_kv_len, head_dim, dtype=torch.float16, device="cuda")
    v_all = torch.randn(batch, kv_heads, max_kv_len, head_dim, dtype=torch.float16, device="cuda")

    # Build cache from first token
    k_init = k_all[:, :, :1, :]
    v_init = v_all[:, :, :1, :]
    cache = SageKVCache.prepare_kv(k_init, v_init, layout=layout, smooth_k=True, quant_granularity="per_warp")

    # Measure latency for each step
    latencies = []
    for kv_len in range(2, max_kv_len + 1):
        q = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")
        k_new = k_all[:, :, kv_len - 1 : kv_len, :]
        v_new = v_all[:, :, kv_len - 1 : kv_len, :]

        cache.append(k_new, v_new)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = sageattn_with_kv_cache(q, cache, is_causal=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

    # Check if growth is closer to linear than quadratic
    # Fit simple trend: compare first half vs second half average latency.
    mid = len(latencies) // 2
    avg_first = sum(latencies[:mid]) / mid
    avg_second = sum(latencies[mid:]) / (len(latencies) - mid)

    print(f"  Average latency first half:  {avg_first*1000:.3f} ms")
    print(f"  Average latency second half: {avg_second*1000:.3f} ms")

    # For linear, ratio should be O(1); for quadratic, it grows significantly.
    # We'll allow up to ~2x as a soft check; much higher suggests degenerate behavior.
    if avg_first > 0 and (avg_second / avg_first) < 2.5:
        print("[PASS] P3-2: Latency growth is consistent with linear scaling")
        return True
    else:
        print("[FAIL] P3-2: Latency growth suggests non-linear or degenerate behavior")
        return False


def test_p3_numerical_consistency():
    """
    Test P3-2: Compare sageattn_with_kv_cache vs sageattn() full recomputation.
    For each kv_len, ensure outputs are allclose.
    """
    print("\n=== Testing P3-2: Numerical consistency vs sageattn() ===")

    batch = 1
    q_heads = 8
    kv_heads = 8
    head_dim = 64
    layout = "HND"

    # Use a fixed set of tokens
    max_kv_len = 32
    k_all = torch.randn(batch, kv_heads, max_kv_len, head_dim, dtype=torch.float16, device="cuda")
    v_all = torch.randn(batch, kv_heads, max_kv_len, head_dim, dtype=torch.float16, device="cuda")

    # Initialize cache with first token
    k_init = k_all[:, :, :1, :]
    v_init = v_all[:, :, :1, :]
    cache = SageKVCache.prepare_kv(k_init, v_init, layout=layout, smooth_k=True, quant_granularity="per_warp")

    all_passed = True

    for kv_len in [4, 8, 16, 32]:
        # Build up cache incrementally
        k_new = k_all[:, :, 1:kv_len, :]
        v_new = v_all[:, :, 1:kv_len, :]
        cache.append(k_new, v_new)

        q = torch.randn(batch, q_heads, 1, head_dim, dtype=torch.float16, device="cuda")

        # sageattn_with_kv_cache result
        out_cache = sageattn_with_kv_cache(q, cache, is_causal=True)

        # sageattn() full recomputation using full history K/V
        k_full = k_all[:, :, :kv_len, :]
        v_full = v_all[:, :, :kv_len, :]
        out_full = sageattn(q, k_full, v_full, tensor_layout=layout, is_causal=True)

        close, msg = allclose_check(out_cache, out_full, atol=2e-2, rtol=2e-2)
        if close:
            print(f"  [PASS] kv_len={kv_len}")
        else:
            print(f"  [FAIL] kv_len={kv_len} - {msg}")
            all_passed = False

    return all_passed


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return

    results = []

    # P3-2: linear growth
    results.append(test_p3_linear_growth())

    # P3-2: numerical consistency vs sageattn()
    results.append(test_p3_numerical_consistency())

    print("\n=== Test Summary ===")
    passed = sum(results) if results else 0
    total = len(results) if results else 0
    print(f"Passed: {passed}/{total}")

    if passed == total and total > 0:
        print("All P3 tests passed!")
    else:
        print("Some P3 tests failed or were skipped.")


if __name__ == "__main__":
    main()
