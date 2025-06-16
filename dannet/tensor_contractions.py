from typing import Callable
import dannet as dt
from dannet import jit
from dannet.lib import core
from dannet.lib.core import TensorLike
from dannet.core import _args_to_tensor

matmul_jit: Callable[
    [core.BaseTensor, core.BaseTensor],
    core.BaseTensor
] = jit(dt.lib.tensor_contractions.matmul)


def matmul(x1: TensorLike, x2: TensorLike) -> dt.lib.core.BaseTensor:
    x1, x2 = _args_to_tensor("matmul", x1, x2)
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

    batch = dt.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    x1 = dt.broadcast_to(x1, batch + x1.shape[-2:])
    x2 = dt.broadcast_to(x2, batch + x2.shape[-2:])

    out = matmul_jit(x1, x2)

    if x1_axis:
        out = dt.squeeze(out, -2)
    if x2_axis:
        out = dt.squeeze(out, -1)
    return out


def outer(x1: TensorLike, x2: TensorLike, /) -> dt.lib.core.BaseTensor:
    x1, x2 = _args_to_tensor("outer", x1, x2)
    return dt.ravel(x1)[:, None] * dt.ravel(x2)[None, :]
