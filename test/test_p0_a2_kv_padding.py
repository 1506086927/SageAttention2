"""
测试脚本：SageAttention P0-A-2 验证 - KV序列维padding删除
验证KV序列维pad删除后，non-causal cross-attn中pad zeros不会污染softmax分母
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

def test_p0_a2_kv_padding_removed():
    """
    Test P0-A-2: KV sequence dimension padding is removed.
    The CUDA kernel should handle tail blocks correctly with apply_out_of_bound_mask.
    """
    print("\n=== Testing P0-A-2: KV padding removed (non-causal cross-attn) ===")
    
    # Test cases where kv_len is not aligned to 32
    test_cases = [
        (1024, 1050),  # Q=1024, KV=1050 (1050 % 32 = 26, k_seq_pad would be 6)
        (512, 530),    # Q=512, KV=530 (530 % 32 = 18, k_seq_pad would be 14)
        (256, 277),    # Q=256, KV=277 (277 % 32 = 21, k_seq_pad would be 11)
    ]
    
    all_passed = True
    for q_len, kv_len in test_cases:
        print(f"\n  Testing Q={q_len}, KV={kv_len}...")
        
        q_shape = (1, 8, q_len, 64)
        kv_shape = (1, 8, kv_len, 64)
        
        q = torch.randn(q_shape, dtype=torch.float16, device='cuda')
        k = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
        v = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
        
        # Reference SDPA (non-causal)
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
            print(f"    [FAIL] sageattn raised exception: {e}")
            all_passed = False
            continue
        
        close, err_msg = allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2)
        if close:
            print(f"    [PASS] Q={q_len}, KV={kv_len}")
        else:
            print(f"    [FAIL] Q={q_len}, KV={kv_len} - {err_msg}")
            print(f"    Max diff: {(out_new - out_ref).abs().max().item()}")
            all_passed = False
    
    return all_passed

def test_p0_a2_short_seq_cross_attn():
    """
    Test P0-A-2 with short sequence cross-attn where seq_q != seq_kv.
    This should be handled by the fallback to PyTorch SDPA (P0-A-1 fix).
    """
    print("\n=== Testing P0-A-2: Short seq cross-attn (seq_q != seq_kv) ===")
    
    test_cases = [
        (16, 77),
        (64, 154),
        (128, 77),
    ]
    
    all_passed = True
    for q_len, kv_len in test_cases:
        print(f"\n  Testing Q={q_len}, KV={kv_len}...")
        
        q_shape = (1, 8, q_len, 64)
        kv_shape = (1, 8, kv_len, 64)
        
        q = torch.randn(q_shape, dtype=torch.float16, device='cuda')
        k = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
        v = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
        
        # Reference SDPA (non-causal)
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
            print(f"    [FAIL] sageattn raised exception: {e}")
            all_passed = False
            continue
        
        close, err_msg = allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2)
        if close:
            print(f"    [PASS] Q={q_len}, KV={kv_len}")
        else:
            print(f"    [FAIL] Q={q_len}, KV={kv_len} - {err_msg}")
            print(f"    Max diff: {(out_new - out_ref).abs().max().item()}")
            all_passed = False
    
    return all_passed

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    
    results = []
    
    # P0-A-2 test: long seq cross-attn with unaligned KV length
    results.append(test_p0_a2_kv_padding_removed())
    
    # P0-A-2 test: short seq cross-attn (seq_q != seq_kv)
    results.append(test_p0_a2_short_seq_cross_attn())
    
    print("\n=== Test Summary ===")
    passed = sum(results) if results else 0
    total = len(results) if results else 0
    print(f"Passed: {passed}/{total}")
    
    if passed == total and total > 0:
        print("All P0-A-2 tests passed!")
    else:
        print("Some P0-A-2 tests failed or were skipped.")

if __name__ == "__main__":
    main()
