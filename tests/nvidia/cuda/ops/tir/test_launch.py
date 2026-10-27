"""Verify host-to-device launch contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import tilefoundry
from tests.nvidia.cuda.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.dsl import *
from tilefoundry.ir.core import Constant, Var, VerifyError
from tilefoundry.ir.tir.launch import Launch
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential
from tilefoundry.ir.tir.symbol_ref import SymbolRef
from tilefoundry.ir.types import Layout
from tilefoundry.ir.types.callable_type import callable_type_for_prim_function
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.verify import verify_module
from tilefoundry_nvidia import CudaTarget


def test_launch_rejects_forwarded_type_mismatch() -> None:
    device_type = Tensor[(8,), 'f32', None, 'gmem']
    host_type = Tensor[(8,), "f32", Layout(shape=(8,), strides=(2,)), "gmem"]
    device_arg = Var(type=device_type, name="device_arg")
    device = PrimFunction(
        name="device",
        params=(device_arg,),
        body=Sequential(body=()),
        target=CudaTarget("nvidia.h200_sxm"),
    )

    host_arg = Var(type=host_type, name="host_arg")
    extent_type = Tensor[(), "i64", None, "rmem"]
    one = Constant(type=extent_type, value=1)
    ref = SymbolRef(name=device.name, type=callable_type_for_prim_function(device))
    launch = Evaluate(callable=Launch(), args=(ref, one, one, one, one, one, one, host_arg))
    host = PrimFunction(
        name="host",
        params=(host_arg,),
        body=Sequential(body=(launch,)),
        target=CpuTarget(),
    )

    with pytest.raises(VerifyError, match=r"forwarded arg\[0\] type"):
        verify_module((host, device))


OPS = Path(__file__).parents[2] / "fixtures" / "ops"
_CUDA = CudaTarget("nvidia.h200_sxm")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_refused_launch_raises() -> None:
    OverAskingLaunch = load_authored_ir(f"{OPS / 'over_asking_launch.py'}:OverAskingLaunch")
    runtime = tilefoundry.compile(OverAskingLaunch, target=_CUDA)
    with pytest.raises(RuntimeError, match=r"tilefoundry: launch of \w+ failed: "):
        runtime(torch.ones(4, 32, device="cuda"))


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_launch_lifts_the_dynamic_shared_memory_default(target: CudaTarget) -> None:
    """An 80 KiB stage runs once its launch asks for 80 KiB and lifts CUDA's 48 KiB default."""
    WideSharedStage = load_authored_ir(f"{OPS / 'wide_shared_stage.py'}:WideSharedStage")
    runtime = tilefoundry.compile(retarget(WideSharedStage, target), target=target)
    torch.manual_seed(5)
    source = torch.randn(128, 160, dtype=torch.float32, device="cuda")
    out = torch.empty_like(source)
    runtime(source, out)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, source, rtol=0, atol=0)
