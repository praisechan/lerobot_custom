import os
from pathlib import Path

from torch.utils.cpp_extension import load


def _cuda_include_dirs():
    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
    ]
    include_dirs = []
    for root in candidates:
        if not root:
            continue
        for include in (
            Path(root) / "include",
            Path(root) / "targets" / "sbsa-linux" / "include",
            Path(root) / "targets" / "x86_64-linux" / "include",
        ):
            if (include / "cuda.h").exists():
                include_dirs.append(str(include))
    return list(dict.fromkeys(include_dirs))


def _cuda_library_flags():
    candidates = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
    ]
    lib_dirs = []
    rpath_dirs = []
    for root in candidates:
        if not root:
            continue
        for libdir in (
            Path(root) / "lib64",
            Path(root) / "targets" / "sbsa-linux" / "lib",
            Path(root) / "targets" / "x86_64-linux" / "lib",
            Path(root) / "lib64" / "stubs",
            Path(root) / "targets" / "sbsa-linux" / "lib" / "stubs",
            Path(root) / "targets" / "x86_64-linux" / "lib" / "stubs",
        ):
            if (libdir / "libcuda.so").exists() or (libdir / "libcudart.so").exists():
                lib_dirs.append(str(libdir))
                if libdir.name != "stubs":
                    rpath_dirs.append(str(libdir))

    flags = [f"-L{p}" for p in dict.fromkeys(lib_dirs)]
    flags.extend(f"-Wl,-rpath,{p}" for p in dict.fromkeys(rpath_dirs))
    flags.append("-lcudart")
    flags.append("-lcuda")
    return flags


def load_greenctx_extension(verbose=None):
    root = Path(__file__).resolve().parent
    if verbose is None:
        verbose = os.environ.get("PI0_GREENCTX_BUILD_VERBOSE", "0") == "1"
    return load(
        name="pi0_greenctx_ext",
        sources=[str(root / "greenctx_helper.cpp")],
        extra_include_paths=_cuda_include_dirs(),
        extra_cflags=["-O3", "-std=c++17"],
        extra_ldflags=_cuda_library_flags(),
        verbose=verbose,
    )
