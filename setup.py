from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "--expt-relaxed-constexpr",
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
    "-gencode", "arch=compute_90,code=sm_90",
    "-std=c++17",
]

cxx_flags = ["-O3", "-std=c++17"]

setup(
    name="cache_evict",
    version="0.1.0",
    description="Fused attention-eviction kernel for bounded-memory long-context inference",
    packages=["cache_evict"],
    ext_modules=[
        CUDAExtension(
            name="cache_evict._C",
            sources=[
                "csrc/bindings.cpp",
                "src/reference_attn.cu",
                "src/fused_attn_evict.cu",
            ],
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch"],
)
