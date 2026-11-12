"""Markers: `gpu` tests are skipped without CUDA, `oracle` tests without the optional FEFF 8.5L library."""
import os
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _feff_library():
    lib = Path(os.environ.get("FEFFGPU_POTPH_LIB", ROOT / "fortran/lib/libfeff8lpotph.so"))
    home = Path(os.environ.get("FEFF_HOME", ROOT / "fortran"))
    return lib.exists() or (home / "lib/libfeff8lpotph.so").exists()


def pytest_collection_modifyitems(config, items):
    no_gpu = pytest.mark.skip(reason="needs a CUDA GPU")
    no_lib = pytest.mark.skip(reason="needs the optional FEFF 8.5L library (fortran/README.md)")
    cuda, lib = torch.cuda.is_available(), _feff_library()
    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(no_gpu)
        if "oracle" in item.keywords and not lib:
            item.add_marker(no_lib)
