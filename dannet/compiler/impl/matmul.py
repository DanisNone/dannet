import math
from typing import Any
import dannet as dt
from dannet.core import Tensor
from dannet.dtypes import DannetDtype
from dannet.gradient import GradientOp
from dannet.typing import DTypeLikeO
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info,
)


def compile_matmul(
    device: dt.Device,
    inputs: tuple[DannetDtype, DannetDtype],
    output: DannetDtype,
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
    )
    return build_program(device, "matmul.cl", build_info).matmul


def matmul_op(x1: Tensor, x2: Tensor, /, dtype: DTypeLikeO = None) -> Tensor:
    if dtype is not None:
        raise NotImplementedError
    device = dt.current_device()

    dt.utils.check_device("matmul", "x1", x1, device)
    dt.utils.check_device("matmul", "x2", x2, device)

    assert x1.ndim >= 2 and x2.ndim >= 2
    assert x1.ndim == x2.ndim
    assert x1.shape[:-2] == x2.shape[:-2]

    M, N_ = x1.shape[-2:]
    N, K = x2.shape[-2:]

    if N != N_:
        # TODO: add message
        raise ValueError
    
    batch = x1.shape[:-2]
    shape = batch + (M, K)
    dtype = dt.promote_types(x1.dtype, x2.dtype)
    out = dt.empty(shape, dtype=dtype, device=device)

    kernel = compile_matmul(
        device,
        inputs=(x1.dtype, x2.dtype),
        output=out.dtype
    )

    # TODO: implement tile-matmul
    event = kernel(
        (math.prod(batch), M, K), None,
        x1._buffer, x2._buffer, out._buffer,
        get_shape_info(device, x1),
        get_shape_info(device, x2),
        get_shape_info(device, out),
    )
    return Tensor(
        out._buffer, out._tensor_info,
        event=event
    )

def matmul_grad(
    grad: Tensor, out: Tensor,
    args: tuple[Tensor, Tensor], kwargs: Any
) -> tuple[Tensor, Tensor]:
    x1, x2, *_ = args

    perm = list(range(x1.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    x1_t = dt.transpose(x1, perm)
    x2_t = dt.transpose(x2, perm)
    
    grad_x = dt.matmul(grad, x2_t)
    grad_y = dt.matmul(x1_t, grad)
    return (grad_x, grad_y)

_matmul = GradientOp(matmul_op, matmul_grad, nondiff_argnum=(2,))

def matmul(x1: dt.typing.TensorLike, x2: dt.typing.TensorLike, /, dtype: DTypeLikeO = None) -> Tensor:
    x1 = dt.array(x1)
    x2 = dt.array(x2)

    if x1.ndim == 0 or x2.ndim == 0:
        raise ValueError(
            'matmul: inputs must be at least 1-dimensional, got scalars'
        )

    x1_axis = False
    if x1.ndim == 1:
        x1_axis = True
        x1 = dt.reshape(x1, (1, -1))

    x2_axis = False
    if x2.ndim == 1:
        x2_axis = True
        x2 = dt.reshape(x2, (-1, 1))

    if x1.shape[-1] != x2.shape[-2]:
        raise ValueError(
            f'matmul: shapes {x1.shape} and {x2.shape} are incompatible: '
            f'last dim of x ({x1.shape[-1]}) must match second last dim '
            f'of y ({x2.shape[-2]})'
        )

    out = _matmul(x1, x2, dtype)

    if x1_axis:
        out = dt.squeeze(out, -2)
    if x2_axis:
        out = dt.squeeze(out, -1)
    return out