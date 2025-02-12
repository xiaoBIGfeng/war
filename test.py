export CUDA_HOME=$(python -c "import torch; print(torch.utils.cpp_extension.CUDA_HOME)")
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

# 动态设置 CUDA_HOME 为 PyTorch 提供的 CUDA 工具链
os.environ["CUDA_HOME"] = CUDA_HOME

setup(
    name='your_extension_name',  # 替换为你的扩展名称
    ext_modules=[
        CUDAExtension(
            'your_extension_name',  # 替换为你的扩展模块名称
            sources=['your_extension_source.cpp', 'your_extension_kernel.cu'],  # 源文件
            extra_compile_args={
                'cxx': ['-O3'],  # 针对 C++ 编译器的优化选项
                'nvcc': ['-O3', '--use_fast_math']  # 针对 CUDA 编译器的选项
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

import torch
import your_extension_name  # 替换为你的扩展名称

print("PyTorch version:", torch.__version__)
print("CUDA version (PyTorch):", torch.version.cuda)
print("Is CUDA available?", torch.cuda.is_available())

