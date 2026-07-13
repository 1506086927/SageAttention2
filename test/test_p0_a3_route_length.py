"""
测试脚本：SageAttention P0-A-3 验证 - 调度只看 Q 长度的问题
验证 Q 很短但 KV 很长时，不会被误判为短路径
"""
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sageattention import sageattn

def allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2):
    """Check if two tensors are close within tolerance."""
    if out_new is None or out_ref is None:
        return False, "One of the outputs is None"
    try:
        return torch.allclose(out_new, out_ref, atol=atol, rtol=rtol), None
    except Exception as e:
        return False, str(e)

def test_p0_a3_route_length_decision():
    """
    Test P0-A-3: Route decision should use max(q_len, kv_len), not just q_len.
    Case: Q=64, KV=4096 (typical SDXL cross-attn bottleneck)
    This should route to long path (qk_int8_pv_fp16_cuda) because kv_len >= 1024.
    """
    print("\n=== Testing P0-A-3: Route length decision (Q short, KV long) ===")
    
    # SDXL cross-attn bottleneck case: Q=64, KV=4096
    q_shape = (1, 8, 64, 64)
    kv_shape = (1, 8, 4096, 64)
    
    q = torch.randn(q_shape, dtype=torch.float16, device='cuda')
    k = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    v = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    
    # Reference SDPA
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None
    )
    
    # SageAttention
    try:
        out_new = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    except Exception as e:
        print(f"[FAIL] sageattn raised exception: {e}")
        return False
    
    close, err_msg = allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2)
    if close:
        print(f"[PASS] P0-A-3: Route length decision test (Q=64, KV=4096)")
        return True
    else:
        print(f"[FAIL] P0-A-3: Route length decision test - {err_msg}")
        print(f"Max diff: {(out_new - out_ref).abs().max().item()}")
        return False

def test_p0_a3_q_short_kv_medium():
    """
    Test P0-A-3: Q=77, KV=4096 cross-attn
    """
    print("\n=== Testing P0-A-3: Q=77, KV=4096 cross-attn ===")
    
    q_shape = (1, 8, 77, 64)
    kv_shape = (1, 8, 4096, 64)
    
    q = torch.randn(q_shape, dtype=torch.float16, device='cuda')
    k = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    v = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    
    # Reference SDPA
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None
    )
    
    # SageAttention
    try:
        out_new = sageattn(q, k, v, tensor_layout="HND", is_causal=False)
    except Exception as e:
        print(f"[FAIL] sageattn raised exception: {e}")
        return False
    
    close, err_msg = allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2)
    if close:
        print(f"[PASS] P0-A-3: Q=77, KV=4096 cross-attn")
        return True
    else:
        print(f"[FAIL] P0-A-3: Q=77, KV=4096 cross-attn - {err_msg}")
        print(f"Max diff: {(out_new - out_ref).abs().max().item()}")
        return False

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    
    results = []
    
    # P0-A-3 test
    results.append(test_p0_a3_route_length_decision())
    results.append(test_p0_a3_q_short_kv_medium())
    
    print("\n=== Test Summary ===")
    passed = sum(results) if results else 0
    total = len(results) if results else 0
    print(f"Passed: {passed}/{total}")
    
    if passed == total and total > 0:
        print("All P0-A-3 tests passed!")
    else:
        print("Some P0-A-3 tests failed or were skipped.")

if __name__ == "__main__":
    main()
