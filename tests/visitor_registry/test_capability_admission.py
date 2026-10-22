"""Module verification refuses an instruction the target's architecture does not list.

See [target §4](docs/spec/target.md#instruction-support-and-execution-resources).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.fixtures.schedule.tir.wgmma_b_nk import gemm as wgmma_gemm
from tests.fixtures.tir.mma import MmHandwritten
from tests.ops.tir.targets import retarget
from tilefoundry.ir.core import Var, VerifyError
from tilefoundry.ir.core.kinds import BinaryKind
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.arith import Binary
from tilefoundry.ir.tir.cpu.tensor import CpuCast1
from tilefoundry.ir.tir.cuda.memory.copy_async_bulk import CopyAsyncBulk
from tilefoundry.ir.tir.cuda.sync.mbarrier import MBarrierArriveExpectTx, MBarrierWaitParity
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, MeshScope, Return, Sequential
from tilefoundry.ir.types import DType, Layout, Mesh, Topology, make_tensor_type
from tilefoundry.target import AmxTarget, CpuTarget, CudaTarget, Target
from tilefoundry.visitor_registry.verify import verify_prim_function

SM89 = CudaTarget("nvidia.rtx_6000_ada")
H200 = CudaTarget("nvidia.h200_sxm")

_SMEM_BAR = make_tensor_type((1,), DType.from_name("i64"), storage="smem")
_PHASE = make_tensor_type((), DType.from_name("i32"), storage="rmem")
_GMEM_TILE = make_tensor_type((32,), DType.f32, storage="gmem")
_GMEM_HALF = make_tensor_type((32,), DType.f16, storage="gmem")
_SMEM_TILE = make_tensor_type((32,), DType.f32, storage="smem")
_RMEM = make_tensor_type((32,), DType.f32, storage="rmem")


def _one_call(op, *types, level: str = "thread", size: int = 32) -> PrimFunction:
    """A function whose whole body is one call of *op* on a one-level mesh."""
    args = tuple(Var(type=t, name=f"a{i}") for i, t in enumerate(types))
    mesh = Mesh((Topology(level, size),), Layout((size,), (1,)))
    return PrimFunction(
        name="fn",
        params=args,
        body=Sequential(
            body=(
                MeshScope(
                    mesh=mesh,
                    binding=Var(name="thread", type=mesh),
                    body=Sequential(body=(Evaluate(callable=op, args=args),)),
                ),
                Return(),
            )
        ),
    )


def _verify(function: PrimFunction, target: Target, topologies=None) -> None:
    """Verify *function* as the entry of a module on *target*."""
    module = Module(
        name="admission",
        functions=(function,),
        entry=function.name,
        target=target,
        topologies=topologies,
    )
    verify_prim_function(function, module_fns=module)


_REFUSED = [
    pytest.param(
        _one_call(MBarrierArriveExpectTx(tx_bytes=16), _SMEM_BAR),
        "mbarrier.expect_tx",
        id="mbarrier_arrive_expect_tx",
    ),
    pytest.param(
        _one_call(MBarrierWaitParity(), _SMEM_BAR, _PHASE),
        "mbarrier.try_wait",
        id="mbarrier_wait_parity",
    ),
    pytest.param(
        _one_call(CopyAsyncBulk(), _GMEM_TILE, _SMEM_TILE, _SMEM_BAR),
        "cp.async.bulk",
        id="copy_async_bulk",
    ),
    pytest.param(wgmma_gemm, "wgmma.mma_async", id="wgmma_atom"),
]


@pytest.mark.parametrize(("function", "capability"), _REFUSED)
def test_an_instruction_the_architecture_lacks_is_refused_naming_its_capability(
    function: PrimFunction, capability: str
) -> None:
    """On sm_89 the call is refused naming what it needs; H200 lists it and admits the same call."""
    with pytest.raises(
        VerifyError, match=f"needs {capability}, which nvidia.rtx_6000_ada does not list"
    ):
        _verify(function, SM89)
    _verify(function, H200)


def test_only_the_selected_atom_is_checked() -> None:
    """The SM80 atom is admitted on sm_89, although the same op also declares WGMMA."""
    module = retarget(MmHandwritten, SM89)
    for function in module.functions:
        verify_prim_function(function, module_fns=module)


def test_cpu_tensor_kernels_need_an_architecture_that_lists_them() -> None:
    """AMX lists cpu_tensor and admits the kernel; a plain CPU target lists nothing and refuses it."""
    cast = _one_call(
        CpuCast1(dtype=DType.f16), _GMEM_TILE, _GMEM_HALF, level="core", size=1
    )
    _verify(cast, AmxTarget(), topologies=(Topology("core", 1),))
    with pytest.raises(VerifyError, match="needs cpu_tensor, which cpu does not list"):
        _verify(cast, CpuTarget())


def test_a_target_neutral_instruction_is_admitted_on_a_plain_cpu_target() -> None:
    """An instruction that declares no capability needs nothing from the architecture."""
    _verify(
        _one_call(Binary(kind=BinaryKind.ADD), _RMEM, _RMEM, _RMEM, size=1), CpuTarget()
    )


def test_without_a_module_there_is_no_target_to_check_against() -> None:
    """A function verified on its own has no target, so its instructions are not checked."""
    verify_prim_function(replace(wgmma_gemm, target=SM89))
