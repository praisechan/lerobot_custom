import os
from pathlib import Path

from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
LEROBOT_ROOT = ROOT.parents[1]
LIBSMCTRL_DIR = LEROBOT_ROOT / "3rdparty" / "BulletServe" / "csrc"


def _cuda_roots():
    roots = [
        os.environ.get("CUDA_HOME"),
        os.environ.get("CUDA_PATH"),
        "/usr/local/cuda",
        "/usr/local/cuda-13.0",
        "/usr/local/cuda-12.8",
        "/usr/local/cuda-12.6",
        "/usr/local/cuda-12.4",
    ]
    return [Path(root) for root in roots if root]


def _cuda_include_dirs():
    include_dirs = []
    for root in _cuda_roots():
        for include in (
            root / "include",
            root / "targets" / "x86_64-linux" / "include",
            root / "targets" / "sbsa-linux" / "include",
        ):
            if (include / "cuda.h").exists():
                include_dirs.append(str(include))
    return list(dict.fromkeys(include_dirs))


def _cuda_library_flags():
    lib_dirs = []
    rpath_dirs = []
    for root in _cuda_roots():
        for libdir in (
            root / "lib64",
            root / "targets" / "x86_64-linux" / "lib",
            root / "targets" / "sbsa-linux" / "lib",
            root / "lib64" / "stubs",
            root / "targets" / "x86_64-linux" / "lib" / "stubs",
            root / "targets" / "sbsa-linux" / "lib" / "stubs",
        ):
            if (libdir / "libcuda.so").exists() or (libdir / "libcudart.so").exists():
                lib_dirs.append(str(libdir))
                if libdir.name != "stubs":
                    rpath_dirs.append(str(libdir))

    flags = [f"-L{path}" for path in dict.fromkeys(lib_dirs)]
    flags.extend(f"-Wl,-rpath,{path}" for path in dict.fromkeys(rpath_dirs))
    flags.extend(["-lcudart", "-lcuda"])
    return flags


def load_libsmctrl_extension(verbose=None):
    if verbose is None:
        verbose = os.environ.get("VLA_LLM_SMCTRL_BUILD_VERBOSE", "0") == "1"
    if not (LIBSMCTRL_DIR / "src" / "libsmctrl.h").exists():
        raise FileNotFoundError(f"libsmctrl sources not found at {LIBSMCTRL_DIR}")

    build_dir = ROOT / "build" / "torch_extensions"
    build_dir.mkdir(parents=True, exist_ok=True)

    return load(
        name="vla_llm_libsmctrl_ext",
        sources=[
            str(ROOT / "csrc" / "smctrl_pybind.cpp"),
            str(LIBSMCTRL_DIR / "src" / "libsmctrl_core.c"),
            str(LIBSMCTRL_DIR / "src" / "libsmctrl_validator.cu"),
        ],
        extra_include_paths=[str(LIBSMCTRL_DIR / "src"), *_cuda_include_dirs()],
        extra_cflags=["-O3", "-std=gnu++17", "-fpermissive"],
        extra_cuda_cflags=["-O3", "--compiler-options", "-fPIC"],
        extra_ldflags=_cuda_library_flags(),
        build_directory=str(build_dir),
        verbose=verbose,
    )
