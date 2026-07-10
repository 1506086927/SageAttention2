"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import subprocess
from packaging.version import parse, Version
from typing import List, Set
from setuptools import setup, Extension
import warnings

from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

HAS_SM75 = False
HAS_SM80 = False
HAS_SM86 = False
HAS_SM89 = False
HAS_SM90 = False
HAS_SM120 = False

SUPPORTED_ARCHS = {"7.5", "8.0", "8.6", "8.9", "9.0", "12.0"}

CXX_FLAGS = ["-g", "-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"]
NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "--use_fast_math",
    "--threads=8",
    "-Xptxas=-v",
    "-diag-suppress=174", 
]

ABI = 1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0
CXX_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]
NVCC_FLAGS += [f"-D_GLIBCXX_USE_CXX11_ABI={ABI}"]

if CUDA_HOME is None:
    raise RuntimeError(
        "Cannot find CUDA_HOME. CUDA must be available to build the package.")

def get_nvcc_cuda_version(cuda_dir: str) -> Version:
    nvcc_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                          universal_newlines=True)
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version

compute_capabilities = set()
device_count = torch.cuda.device_count()
for i in range(device_count):
    major, minor = torch.cuda.get_device_capability(i)
    if major < 7:
        warnings.warn(f"skipping GPU {i} with compute capability {major}.{minor}")
        continue
    compute_capabilities.add(f"{major}.{minor}")

nvcc_cuda_version = get_nvcc_cuda_version(CUDA_HOME)
if not compute_capabilities:
    raise RuntimeError("No GPUs found. Please specify the target GPU architectures or build on a machine with GPUs.")
else:
    print(f"Detect GPUs with compute capabilities: {compute_capabilities}")

for capability in compute_capabilities:
    if capability.startswith("8.0"):
        HAS_SM80 = True
        num = "80"
    elif capability.startswith("7.5"):
        HAS_SM75 = True
        num = "75"
    elif capability.startswith("8.6"):
        HAS_SM86 = True
        num = "86"
    elif capability.startswith("8.9"):
        HAS_SM89 = True
        num = "89"
    elif capability.startswith("9.0"):
        HAS_SM90 = True
        num = "90a" 
    elif capability.startswith("12.0"):
        HAS_SM120 = True
        num = "120" 
    NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=sm_{num}"]
    if capability.endswith("+PTX"):
        NVCC_FLAGS += ["-gencode", f"arch=compute_{num},code=compute_{num}"]

ext_modules = []

if HAS_SM80 or HAS_SM75 or HAS_SM86 or HAS_SM89 or HAS_SM90 or HAS_SM120:
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm80",
        sources=[
            "csrc/qattn/pybind_sm80.cpp",
            "csrc/qattn/qk_int_sv_f16_cuda_sm80.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_extension)

if HAS_SM89 or HAS_SM120:
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm89",
        sources=[
            "csrc/qattn/pybind_sm89.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm89.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(qattn_extension)

if HAS_SM90:
    qattn_extension = CUDAExtension(
        name="sageattention._qattn_sm90",
        sources=[
            "csrc/qattn/pybind_sm90.cpp",
            "csrc/qattn/qk_int_sv_f8_cuda_sm90.cu",
        ],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
        extra_link_args=['-lcuda'],
    )
    ext_modules.append(qattn_extension)

fused_extension = CUDAExtension(
    name="sageattention._fused",
    sources=["csrc/fused/pybind.cpp", "csrc/fused/fused.cu"],
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS,
    },
)
ext_modules.append(fused_extension)

if HAS_SM75:
    fast_attn_sources = ["csrc/fast_attn_sm75.cu", "csrc/fast_attn_sm75_bind.cpp"]
else:
    fast_attn_sources = ["csrc/fast_attn.cpp"]
    
fast_attn_extension = CUDAExtension(
    name="sageattention._fast_attn",
    sources=fast_attn_sources,
    extra_compile_args={
        "cxx": CXX_FLAGS,
        "nvcc": NVCC_FLAGS if HAS_SM75 else [],
    },
)
ext_modules.append(fast_attn_extension)

if HAS_SM75:
    sm75_fast_dispatch_extension = CUDAExtension(
        name="sageattention._sm75_fast_dispatch",
        sources=["csrc/sm75_fast_dispatch.cu", "csrc/sm75_fast_dispatch_bind.cpp"],
        extra_compile_args={
            "cxx": CXX_FLAGS,
            "nvcc": NVCC_FLAGS,
        },
    )
    ext_modules.append(sm75_fast_dispatch_extension)

setup(
    name='sageattention', 
    version='2.1.2',
    author='SageAttention team',
    license='Apache 2.0 License',  
    description='Accurate and efficient plug-and-play low-bit attention, modified for Turing.',  
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown', 
    url='https://github.com/thu-ml/SageAttention', 
    packages=find_packages(),
    python_requires='>=3.9',
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
