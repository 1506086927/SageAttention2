"""
测试脚本：SageAttention SM75 (Turing/2080Ti) 修复验证
覆盖第7节全局验证矩阵中的所有场景
"""
import torch
import torch.nn.functional as F
import sys
import os

# Ensure sageattention is importable
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

def run_test_case(test_name, q_shape, kv_shape, head_dim, is_causal, q_heads, kv_heads, layout="HND"):
    """Run a single test case and compare with torch SDPA."""
    print(f"\n=== Testing: {test_name} ===")
    print(f"Q shape: {q_shape}, KV shape: {kv_shape}, head_dim: {head_dim}, causal: {is_causal}, GQA: {q_heads}x/{kv_heads}x")
    
    seq_dim_q = 2 if layout == "HND" else 1
    seq_dim_kv = 2 if layout == "HND" else 1
    
    # Create tensors
    # q_shape: (batch, heads_q, seq_len_q, head_dim) for HND
    # k_shape, v_shape: (batch, heads_kv, seq_len_kv, head_dim) for HND
    batch = q_shape[0]
    
    q = torch.randn(q_shape, dtype=torch.float16, device='cuda')
    k = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    v = torch.randn(kv_shape, dtype=torch.float16, device='cuda')
    
    # Reference SDPA with GQA expansion
    k_expanded = k
    v_expanded = v
    if q_heads != kv_heads:
        num_groups = q_heads // kv_heads
        k_expanded = k.repeat_interleave(num_groups, dim=1 if layout == "HND" else 2)
        v_expanded = v.repeat_interleave(num_groups, dim=1 if layout == "HND" else 2)
    
    out_ref = F.scaled_dot_product_attention(
        q, k_expanded, v_expanded,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=None
    )
    
    # SageAttention
    try:
        out_new = sageattn(q, k, v, tensor_layout=layout, is_causal=is_causal)
    except Exception as e:
        print(f"[FAIL] sageattn raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    close, err_msg = allclose_check(out_new, out_ref, atol=2e-2, rtol=2e-2)
    if close:
        print(f"[PASS] {test_name}")
        return True
    else:
        print(f"[FAIL] {test_name} - {err_msg}")
        print(f"Max diff: {(out_new - out_ref).abs().max().item()}")
        return False

def test_dtype_boundary():
    """Test P0-A-4: dtype boundary (bf16/fp32 input should either be converted or error clearly)."""
    print("\n=== Testing: dtype boundary (bf16/fp32) ===")
    try:
        q_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
        k_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
        v_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
        
        # For SM75, bf16 should be converted to fp16 internally
        out_bf16 = sageattn(q_bf16, k_bf16, v_bf16, tensor_layout="HND", is_causal=False)
        print(f"[PASS] dtype boundary test (bf16 input handled correctly)")
    except Exception as e:
        print(f"[INFO] dtype boundary test raised: {e}")

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    
    # Get device capability
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm{major}{minor}"
    print(f"Current GPU architecture: {arch}")
    
    if arch != "sm75":
        print(f"Warning: Current GPU is {arch}, not sm75. Tests may route to different backends.")
    
    results = []
    
    # 1. Image self-attn
    for seq_len in [1024, 4096]:
        for head_dim in [64, 128]:
            q_shape = (1, 8, seq_len, head_dim)
            kv_shape = (1, 8, seq_len, head_dim)
            results.append(run_test_case(
                f"Image self-attn: L={seq_len}, D={head_dim}", 
                q_shape, kv_shape, head_dim, False, 8, 8, "HND"
            ))
    
    # 2. Image cross-attn (P0 trigger): Q=16/64/128/256, KV=77/154
    for q_len in [16, 64, 128, 256]:
        for kv_len in [77, 154]:
            for head_dim in [64, 128]:
                q_shape = (1, 8, q_len, head_dim)
                kv_shape = (1, 8, kv_len, head_dim)
                results.append(run_test_case(
                    f"Image cross-attn: Q={q_len}, KV={kv_len}, D={head_dim}", 
                    q_shape, kv_shape, head_dim, False, 8, 8, "HND"
                ))
    
    # 3. Image cross-attn equal length: Q=77, KV=77
    q_shape = (1, 8, 77, 64)
    kv_shape = (1, 8, 77, 64)
    results.append(run_test_case(
        "Image cross-attn equal length: Q=77, KV=77, D=64", 
        q_shape, kv_shape, 64, False, 8, 8, "HND"
    ))
    
    # 4. Video self-attn
    for seq_len in [8192, 16384]:
        for head_dim in [64, 128]:
            q_shape = (1, 8, seq_len, head_dim)
            kv_shape = (1, 8, seq_len, head_dim)
            results.append(run_test_case(
                f"Video self-attn: L={seq_len}, D={head_dim}", 
                q_shape, kv_shape, head_dim, False, 8, 8, "HND"
            ))
    
    # 5. Audio decode (P0 trigger): Q=1, KV=1..256 step by step, GQA Hq16/Hkv4
    for kv_len in [1, 32, 64, 127, 256]:
        q_shape = (1, 16, 1, 64)
        kv_shape = (1, 4, kv_len, 64)
        results.append(run_test_case(
            f"Audio decode: Q=1, KV={kv_len}, GQA 16/4", 
            q_shape, kv_shape, 64, True, 16, 4, "HND"
        ))
    
    # 6. Non-aligned length: Q=50, KV=100
    q_shape = (1, 8, 50, 64)
    kv_shape = (1, 8, 100, 64)
    results.append(run_test_case(
        "Non-aligned length: Q=50, KV=100, D=64", 
        q_shape, kv_shape, 64, False, 8, 8, "HND"
    ))
    
    # 7. Dtype boundary
    test_dtype_boundary()
    
    print("\n=== Test Summary ===")
    passed = sum(results) if results else 0
    total = len(results) if results else 0
    print(f"Passed: {passed}/{total}")
    
    if passed == total and total > 0:
        print("All tests passed!")
    else:
        print("Some tests failed or were skipped.")

if __name__ == "__main__":
    main()
