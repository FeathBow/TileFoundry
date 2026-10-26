"""Device entries with no host entry: single kernels and dispatch prototypes."""

from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_nvidia import CudaTarget

_CUDA = CudaTarget("nvidia.h200_sxm")
_S = DimVar("S", 1, 255)


def _rows(threads: int) -> ShardLayout:
    """One element per thread of a *threads*-wide block."""
    return ShardLayout(
        layout=Layout(shape=(threads,), strides=(1,)),
        attrs=(S(0),),
        mesh=Mesh(topologies=(Topology("thread", threads),), layout=Layout((threads,), (1,))),
    )


@module(entry="copy_one", target=_CUDA, topologies=(Topology("thread", 128),))
class _OneKernel:
    """A single kernel, and no host entry to call it."""

    @prim_func(target=_CUDA)
    def copy_one(x: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128))
            T.copy(view, view)
            T.sync(thread)


@module(entry="square", target=_CUDA, topologies=(Topology("thread", 128),))
class _Prototype:
    """A prototype with no body of its own, and two variants that have one."""

    @prim_func(target=_CUDA)
    def square(x: Tensor[(_S,), "f32"]):
        pass

    @square.specialize(RangePattern("S", 1, 127))
    def small(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            T.copy(view, view)
            T.sync(thread)

    @square.specialize(RangePattern("S", 128, 255))
    def large(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            T.copy(view, view)
            T.sync(thread)


@module(entry="stage_twice", target=_CUDA, topologies=(Topology("thread", 128),))
class _SharedRing:
    """A kernel staging through two shared-memory slots; the second ends at byte 1024."""

    @prim_func(target=_CUDA)
    def stage_twice(x: Tensor[(128,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128))
            low = T.tensor_view(0, dtype="f32", storage="smem", layout=_rows(128), shape=(128,))
            high = T.tensor_view(512, dtype="f32", storage="smem", layout=_rows(128), shape=(128,))
            T.copy(view, low)
            T.sync(thread)
            T.copy(low, view)
            T.copy(view, high)
            T.sync(thread)
            T.copy(high, view)


@module(entry="stage", target=_CUDA, topologies=(Topology("thread", 128),))
class _StagedPrototype:
    """A prototype whose variants stage through different slots; the farther ends at byte 2560."""

    @prim_func(target=_CUDA)
    def stage(x: Tensor[(_S,), "f32"]):
        pass

    @stage.specialize(RangePattern("S", 1, 127))
    def near(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            slot = T.tensor_view(0, dtype="f32", storage="smem", layout=_rows(128), shape=(128,))
            T.copy(view, slot)
            T.sync(thread)
            T.copy(slot, view)

    @stage.specialize(RangePattern("S", 128, 255))
    def far(x: Tensor[(_S,), "f32"]):
        with Mesh((Topology("thread", 128),), Layout((128,), (1,))) as thread:
            view = T.tensor_view(T.ptr_of(x), layout=_rows(128), shape=(128,))
            slot = T.tensor_view(2048, dtype="f32", storage="smem", layout=_rows(128), shape=(128,))
            T.copy(view, slot)
            T.sync(thread)
            T.copy(slot, view)


@module(
    entry="stage_halves",
    target=_CUDA,
    topologies=(Topology("cta", 2), Topology("thread", 128)),
)
class _CtaHalves:
    """Two CTAs, each declaring its 128 KiB half of a (65536,) f32 view in shared memory."""

    @prim_func(target=_CUDA)
    def stage_halves(x: Tensor[(128,), "f32"]):
        with Mesh((Topology("cta", 2),), Layout((2,), (1,))) as cta:
            halves = T.tensor_view(  # noqa: F841
                0,
                dtype="f32",
                storage="smem",
                layout=ShardLayout(Layout((65536,), (1,)), (S(0),), cta),
                shape=(65536,),
            )
