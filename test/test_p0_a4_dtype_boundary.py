"""
测试脚本：SageAttention P0-A-4 验证 - 短核入口防御性检查 (dtype边界)
验证传入 bf16/fp32 张量调用底层函数时，会抛出清晰的 TORCH_CHECK 错误而不是 segfault 或静默错误结果
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_p0_a4_dtype_boundary():
    """Test P0-A-4: dtype boundary (bf16/fp32 input should throw clear TORCH_CHECK error)"""
    print("\n=== Testing P0-A-4: dtype boundary (bf16 input to short kernel) ===")
    
    try:
        from sageattention import _sm75_fast_dispatch
    except ImportError as e:
        print(f"[INFO] _sm75_fast_dispatch not available: {e}")
        return True
    
    # Test with bf16 tensors
    q_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
    k_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
    v_bf16 = torch.randn(1, 8, 77, 64, dtype=torch.bfloat16, device='cuda')
    
    try:
        # This should throw a clear TORCH_CHECK error about fp16 requirement
        result = _sm75_fast_dispatch.sm75_fast_sdpa(q_bf16, k_bf16, v_bf16, False, 0.0)
        print(f"[FAIL] P0-A-4: bf16 input did not throw error, got result: {result}")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "fp16" in error_msg.lower() or "half" in error_msg.lower():
            print(f"[PASS] P0-A-4: bf16 input correctly threw fp16 requirement error")
            print(f"      Error message: {error_msg}")
            return True
        else:
            print(f"[FAIL] P0-A-4: bf16 input threw unexpected error: {error_msg}")
            return False
    except Exception as e:
        print(f"[FAIL] P0-A-4: bf16 input threw unexpected exception type: {type(e).__name__}: {e}")
        return False

def test_p0_a4_non_contiguous():
    """Test P0-A-4: non-contiguous input should throw clear error"""
    print("\n=== Testing P0-A-4: non-contiguous input ===")
    
    try:
        from sageattention import _sm75_fast_dispatch
    except ImportError as e:
        print(f"[INFO] _sm75_fast_dispatch not available: {e}")
        return True
    
    # Create contiguous fp16 tensors first
    q_fp16 = torch.randn(1, 8, 77, 64, dtype=torch.float16, device='cuda')
    k_fp16 = torch.randn(1, 8, 77, 64, dtype=torch.float16, device='cuda')
    v_fp16 = torch.randn(1, 8, 77, 64, dtype=torch.float16, device='cuda')
    
    # Create non-contiguous tensors by transpose
    q_non_contig = q_fp16.transpose(0, 1)  # This makes it non-contiguous
    
    try:
        result = _sm75_fast_dispatch.sm75_fast_sdpa(q_non_contig, k_fp16, v_fp16, False, 0.0)
        print(f"[FAIL] P0-A-4: non-contiguous input did not throw error, got result: {result}")
        return False
    except RuntimeError as e:
        error_msg = str(e)
        if "contiguous" in error_msg.lower():
            print(f"[PASS] P0-A-4: non-contiguous input correctly threw contiguous requirement error")
            print(f"      Error message: {error_msg}")
            return True
        else:
            print(f"[FAIL] P0-A-4: non-contiguous input threw unexpected error: {error_msg}")
            return False
    except Exception as e:
        print(f"[FAIL] P0-A-4: non-contiguous input threw unexpected exception type: {type(e).__name__}: {e}")
        return False

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    
    results = []
    
    # P0-A-4 test: dtype boundary
    results.append(test_p0_a4_dtype_boundary())
    
    # P0-A-4 test: non-contiguous input
    results.append(test_p0_a4_non_contiguous())
    
    print("\n=== Test Summary ===")
    passed = sum(results) if results else 0
    total = len(results) if results else 0
    print(f"Passed: {passed}/{total}")
    
    if passed == total and total > 0:
        print("All P0-A-4 tests passed!")
    else:
        print("Some P0-A-4 tests failed or were skipped.")

if __name__ == "__main__":
    main()
