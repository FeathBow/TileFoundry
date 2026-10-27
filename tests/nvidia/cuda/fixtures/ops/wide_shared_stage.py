"""A tile staged through more dynamic shared memory than CUDA grants without opting in."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")


def _rows(mesh) -> ShardLayout:
    """One contiguous 160-element row per thread."""
    return ShardLayout(Layout((128, 160), (160, 1)), (S(0),), mesh)


@module(entry="wide_stage", target=_CUDA, topologies=(Topology("thread", 128),))
class WideSharedStage:
    """One block stages a (128, 160) f32 tile, 80 KiB, through shared memory and back.

    The module has no host entry, so the synthesized one decides what the launch asks for.
    """

    @prim_func(target=_CUDA)
    def wide_stage(x: Tensor[(128, 160), "f32"], out: Tensor[(128, 160), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            source = T.tensor_view(T.ptr_of(x), layout=_rows(thread))
            stage = T.tensor_view(
                0, dtype="f32", storage="smem", layout=_rows(thread), shape=(128, 160)
            )
            target = T.tensor_view(T.ptr_of(out), layout=_rows(thread))
            T.copy(source, stage)
            T.sync(thread)
            T.copy(stage, target)
