
**对于 <= Torch 2.10 **：如果由于某些依赖限制你必须使用这个环境，那么手动去改一下** `python\Lib\site-packages\torch\include\torch\csrc\dynamo\compiled_autograd.h`  

### 修改前（原代码）：
    } else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    }

### 修改后：
    }/* else if constexpr (::std::is_same_v<T, ::std::string>) {
      return at::StringType::get();
    } */


Sage attention hacked for NVIDIA Turing GPUs. See the real repo: https://github.com/thu-ml/SageAttention


MMA "fixed" thanks to https://github.com/mit-han-lab/nunchaku

qattn outputs are low quality but only tested SDXL
sparge attention can, in theory, run given the same treatment


------------------
Status as of 2.1.1:

Compiles on cuda 11.8

fused kernel : working on SM75

qattn: compiles and runs when selected (nans)

9/1/25 - triton w/fused works on ComfyUI with SageAttention command line.
