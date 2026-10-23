"""A tile GEMM staged through shared memory, accumulated over a K loop.

See [runtime §2.6](docs/spec/runtime.md#26-cudaops).
"""

from __future__ import annotations

import pytest
import torch

import tilefoundry
from tests.ops.tir.targets import CUDA_TARGETS, retarget
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget, CudaTarget
from tilefoundry.visitor_registry.verify import verify_module

_CUDA = CudaTarget("nvidia.h200_sxm")
_MESH_LAYOUT = Layout(shape=(4, 8), strides=(1, 4))


@module(entry="gemm_host", target=_CUDA, topologies=(Topology("thread", 32),))
class Fp8Gemm:
    """One warp computes a 16x8 F32 tile of A (16, 128) by B (8, 128) transposed, in E4M3.

    It implements ``tests/fixtures/schedule/tir/sm89_e4m3_gemm.py``: both staging
    tiles use 512 bytes of shared memory at offset 0, and each K step stages A with
    ``cp.async`` and gathers its fragment, then does the same for B, then the MMA.
    The waits and barriers between those steps are this implementation's.
    """

    @prim_func(target=_CUDA)
    def gemm_device(
        a: Tensor[(16, 128), "fp8e4m3"],
        b: Tensor[(8, 128), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        with Mesh((Topology("thread", 32),), _MESH_LAYOUT, names=("warp", "lane")) as m:
            a_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 32),
                    "fp8e4m3",
                    ((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1)),
                    "rmem",
                ]
            )
            b_frag = T.alloc_tensor(
                tensor_type=Tensor[
                    (32, 8),
                    "fp8e4m3",
                    ((2, 4 @ m.warp, 4, 8 @ m.lane), (128, 32, 8, 1)),
                    "rmem",
                ]
            )
            acc = T.alloc_tensor(
                tensor_type=Tensor[
                    (16, 8), "f32", ((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1)), "rmem"
                ]
            )
            T.fill(acc, 0.0)
            for k in range(0, 128, 32):
                a_slice = T.tensor_view(
                    T.ptr_of(a[0:16, k:k + 32]),
                    shape=(2, 8, 4, 8),
                    layout=((2, 8 @ m.lane, 4 @ m.warp, 8), (1024, 128, 8, 1)),
                )
                a_staged = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    shape=(2, 8, 4, 8),
                    layout=((2, 8 @ m.lane, 4 @ m.warp, 8), (256, 32, 8, 1)),
                )
                T.copy_async(a_slice, a_staged)
                T.cp_async_commit()
                T.cp_async_wait(n=0)
                T.sync(m)
                a_operand = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    shape=(16, 32),
                    layout=((2, 8 @ m.lane, 2, 4 @ m.warp, 4), (256, 32, 16, 4, 1)),
                )
                T.copy(a_operand, a_frag)
                T.sync(m)
                b_slice = T.tensor_view(
                    T.ptr_of(b[0:8, k:k + 32]),
                    shape=(8, 4, 8),
                    layout=((8 @ m.lane, 4 @ m.warp, 8), (128, 8, 1)),
                )
                b_staged = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    shape=(8, 4, 8),
                    layout=((8 @ m.lane, 4 @ m.warp, 8), (32, 8, 1)),
                )
                T.copy_async(b_slice, b_staged)
                T.cp_async_commit()
                T.cp_async_wait(n=0)
                T.sync(m)
                b_operand = T.tensor_view(
                    0,
                    dtype="fp8e4m3",
                    storage="smem",
                    shape=(32, 8),
                    layout=((2, 4 @ m.warp, 4, 8 @ m.lane), (16, 4, 1, 32)),
                )
                T.copy(b_operand, b_frag)
                T.tiled_mma(acc, a_frag, b_frag, atom=T.cuda.sm89.Mma())
                T.sync(m)
            c_view = T.tensor_view(
                T.ptr_of(c), layout=((2, 8 @ m.lane, 4 @ m.warp, 2), (64, 8, 2, 1))
            )
            T.copy(acc, c_view)

    @prim_func(target=CpuTarget())
    def gemm_host(
        a: Tensor[(16, 128), "fp8e4m3"],
        b: Tensor[(8, 128), "fp8e4m3"],
        c: Tensor[(16, 8), "f32"],
    ):
        launch(gemm_device, a, b, c, grid=(1, 1, 1), block=(32, 1, 1), dynamic_smem=512)


@pytest.mark.parametrize("target", CUDA_TARGETS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_e4m3_gemm_through_shared_memory_equals_the_f32_product(target: CudaTarget) -> None:
    """Integer inputs keep every partial sum exact, so the tile must equal the F32 product."""
    module = retarget(Fp8Gemm, target)
    verify_module(module)
    runtime = tilefoundry.compile(module, target=target)
    torch.manual_seed(0)
    a = torch.randint(-4, 5, (16, 128), device="cuda").float().to(torch.float8_e4m3fn)
    b = torch.randint(-4, 5, (8, 128), device="cuda").float().to(torch.float8_e4m3fn)
    c = torch.empty(16, 8, dtype=torch.float32, device="cuda")
    runtime(a, b, c)
    torch.cuda.synchronize()
    assert torch.equal(c, a.float() @ b.float().T)
