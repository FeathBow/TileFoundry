"""Verify the operand forms an effect Op accepts."""

from __future__ import annotations

import pytest

from tilefoundry.ir.core import Call, Constant, Tuple, Var, VerifyError
from tilefoundry.ir.hir.tensor.slice import Slice
from tilefoundry.ir.hir.tensor.tuple_get_item import TupleGetItem
from tilefoundry.ir.tir.memory import Copy, TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, LetStmt, Sequential
from tilefoundry.ir.types import DType, PointerType, TensorType, TupleType, make_tensor_type
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.verify import verify_prim_function


@pytest.mark.parametrize("kind", ("TupleGetItem", "TensorView", "Slice"))
def test_evaluate_refuses_an_unbound_view(kind: str) -> None:
    """A ring stage, a pointer view or a window reaches an instruction only through its binding."""
    stage = make_tensor_type((8,), DType.f32, storage="smem")
    stages = Var(type=TupleType((stage, stage)), name="stages")
    ring = Tuple(
        tuple(
            Call(
                TensorView(dtype="f32", storage=StorageKind.SMEM, layout=stage.layout, shape=(8,)),
                (Constant(offset, type=PointerType(DType.f32, StorageKind.SMEM)),),
                type=stage,
            )
            for offset in (0, 32)
        ),
        type=stages.type,
    )
    source = Var(type=make_tensor_type((8,), DType.f32, storage="gmem"), name="source")
    zero = Constant(0, type=TensorType.scalar(DType.i64))
    operand = {
        "TupleGetItem": Call(TupleGetItem(), (stages, zero), type=stage),
        "TensorView": ring.elements[1],
        "Slice": Call(
            Slice(sizes=(8,), strides=(1,)),
            (source, Tuple((zero,), type=TupleType((zero.type,)))),
            type=source.type,
        ),
    }[kind]
    function = PrimFunction(
        "fill_stage",
        (source,),
        Sequential((LetStmt(stages, ring), Evaluate(Copy(), (source, operand)))),
    )
    with pytest.raises(VerifyError, match=rf"Copy: arg\[1\] is a {kind} call"):
        verify_prim_function(function)
