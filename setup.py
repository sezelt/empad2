from setuptools import setup, Extension

# from distutils.core import setup
# from distutils.extension import Extension
from Cython.Build import cythonize
import platform
import os
from glob import glob
import numpy as np

if platform.system() == "Windows":
    extra_compile_args = ["/openmp"]
    extra_link_args = ["/openmp"]
elif platform.system() == "Darwin":
    extra_link_args = ["-fopenmp"]
    extra_compile_args = ["-fopenmp"]

    # we are on a Mac, link to the Homebrew installation of llvm
    # extra_link_args.append(
    #     "-Wl,-rpath," + glob("/usr/local/Cellar/llvm/*/lib/clang/*/include/")[0]
    # )
    extra_link_args.append("-Wl,-rpath," + glob("/usr/local/opt/gcc/lib/gcc/9/")[0])
    # extra_link_args.append("-L/usr/local/opt/gcc/lib/gcc/9/")

    # Previously, I used Homebrew-installed gcc...
    # However, it has been giving me unexpected behavior in parallel code
    # os.environ["CC"] = "gcc-9"

    # LLVM Clang seems to work correctly!
    # If CC and LDSHARED are not set in the envoronment, try to find Homebrew LLVM clang...
    if "CC" not in os.environ:
        os.environ["CC"] = glob("/usr/local/Cellar/llvm/9*/bin/clang")[0]
    # if "LDSHARED" not in os.environ:
    #     os.environ["LDSHARED"] = (
    #         glob("/usr/local/Cellar/llvm/9*/bin/clang")[0] + " -bundle"
    #     )
else:
    # if we are on linux, check if we are using intel or Cray/gcc
    if "icc" in os.environ.get("CC",""):
        extra_compile_args = ["-qopenmp"]
        extra_link_args = ["-qopenmp"]
    else:
        extra_compile_args = ["-fopenmp"]
        extra_link_args = ["-fopenmp"]

if platform.system() != "Windows":
    # extra_compile_args.append("-O3")
    extra_compile_args.append("-march=native")

extra_compile_args.append("-O3")

ext_modules = [
    Extension(
        "empad2.combine",
        ["src/empad2/combine.pyx"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        include_dirs=[np.get_include()],
    ),
]

setup(
    name="empad2",
    ext_modules=cythonize(ext_modules, compiler_directives={"language_level": "3"}),
)
