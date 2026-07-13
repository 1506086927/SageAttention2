"""
测试脚本：SageAttention P0-A-2 与 P0-A-6 验证
P0-A-2: 删除 Python 层序列维 padding 对 softmax 的污染 (KV 维 pad 导致 softmax 分母污染)
P0-A-6: 修复拼写/异常捕获 bug (裸 except 替换为 except (ImportError, OSError), _is_caual -> _is_causal)
"""
import torch
import torch.nn.functional as F
import sys
import os
import re

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

def test_p0_a2_kv_padding_pollution():
    """
    Test P0-A-2: Non-causal cross-attention with long sequence where KV length is not aligned to 32.
    This should trigger sageattn_qk_int8_pv_fp16_cuda (seq_len >= 1024), 
    and the KV padding zeros should NOT pollute softmax denominator.
    """
    print("\n=== Testing P0-A-2: KV padding pollution (long seq cross-attn) ===")
    
    # Q=1024, KV=1050 (1050 % 32 = 26, so k_seq_pad = 6)
    # non-causal cross-attn
    q_shape = (1, 8, 1024, 64)
    kv_shape = (1, 8, 1050, 64)
    
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
        print(f"[PASS] P0-A-2: KV padding pollution test")
        return True
    else:
        print(f"[FAIL] P0-A-2: KV padding pollution test - {err_msg}")
        print(f"Max diff: {(out_new - out_ref).abs().max().item()}")
        return False

def test_p0_a6_no_naked_except():
    """Test P0-A-6: No naked 'except:' in sageattention/"""
    print("\n=== Testing P0-A-6: No naked except: ===")
    core_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sageattention', 'core.py')
    with open(core_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find naked except: (except: not followed by Exception or specific exception type)
    # Pattern: 'except:\n' or 'except :' or 'except \n'
    naked_except_pattern = re.compile(r'^except\s*:(?!\s*(Exception|ImportError|OSError|KeyboardInterrupt|SystemExit|\s*#))', re.MULTILINE)
    matches = naked_except_pattern.findall(content)
    
    if matches:
        print(f"[FAIL] P0-A-6: Found naked except: in core.py")
        return False
    else:
        print(f"[PASS] P0-A-6: No naked except: found in core.py")
        return True

def test_p0_a6_no_is_caual_typo():
    """Test P0-A-6: No _is_caual typo, should be _is_causal"""
    print("\n=== Testing P0-A-6: No _is_caual typo ===")
    core_py_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sageattention', 'core.py')
    with open(core_py_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if '_is_caual' in content:
        print(f"[FAIL] P0-A-6: Found _is_caual typo in core.py")
        return False
    else:
        print(f"[PASS] P0-A-6: No _is_caual typo found in core.py")
        return True

def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests.")
        return
    
    results = []
    
    # P0-A-2 test
    results.append(test_p0_a2_kv_padding_pollution())
    
    # P0-A-6 tests
    results.append(test_p0_a6_no_naked_except())
    results.append(test_p0_a6_no_is_caual_typo())
    
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
