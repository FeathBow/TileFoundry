"""What ``schedule finalize`` writes, compiled for its CUDA target."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import tilefoundry
from tests.nvidia.cuda.schedule._support import _finalized
from tilefoundry.cli.source import load_authored_ir
from tilefoundry_nvidia import CudaTarget

HIR = Path(__file__).parents[1] / "fixtures" / "schedule" / "hir"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="compiling CUDA needs nvcc")
def test_whole_register_windows_compile() -> None:
    """The sm_89 GEMM's MMA reads whole register tiles, and its output the whole accumulator."""
    module = load_authored_ir(f"{HIR / 'sm89_e4m3_gemm.py'}:SM89_E4M3_GEMM")
    tilefoundry.compile(_finalized(module), target=CudaTarget("nvidia.rtx_6000_ada"))
