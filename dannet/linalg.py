import numpy as np
import dannet
from dannet import lib
from dannet.lib import core, dtypes
from dannet.lib.typing import Axis, TensorLike


def norm(
    x: TensorLike, ord: int | str | None = None,
    axis: Axis = None,
    keepdims: bool = False
) -> core.BaseTensor:
    x, = dannet.core._args_to_tensor("linalg.norm", x)
    x = x.astype(lib.dtypes.promote_to_inexact(x.dtype))
    ndim = x.ndim

    if axis is None:
        if ord is None:
            return dannet.sqrt(
                dannet.sum(
                    dannet.real(x * x.conj()),
                    keepdims=keepdims
                )
            )
        axis = tuple(range(ndim))
    axis = lib.utils.normalize_axis_tuple(axis, ndim, "x")
    num_axes = len(axis)
    if num_axes == 1:
        return vector_norm(
            x, ord=2 if ord is None else ord,
            axis=axis, keepdims=keepdims
        )
    elif num_axes == 2:
        row_axis, col_axis = axis
        if ord is None or ord in ('f', 'fro'):
            return dannet.sqrt(
                dannet.sum(
                    dannet.real(x * x.conj()),
                    axis=axis,
                    keepdims=keepdims
                )
            )
        elif ord == 1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1

            s = dannet.sum(dannet.abs(x), axis=row_axis, keepdims=keepdims)
            return dannet.max(s, axis=col_axis, keepdims=keepdims, initial=0)
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            s = dannet.sum(dannet.abs(x), axis=row_axis, keepdims=keepdims)
            return dannet.min(s, axis=col_axis, keepdims=keepdims)
        elif ord == np.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            s = dannet.sum(dannet.abs(x), axis=col_axis, keepdims=keepdims)
            return dannet.max(s, axis=row_axis, keepdims=keepdims, initial=0)
        elif ord == -np.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            s = dannet.sum(dannet.abs(x), axis=col_axis, keepdims=keepdims)
            return dannet.min(s, axis=row_axis, keepdims=keepdims)
        elif ord in ('nuc', 2, -2):
            raise NotImplementedError
        else:
            raise ValueError(f"Invalid order '{ord}' for matrix norm.")
    else:
        raise ValueError(
            f"Improper number of axes for norm: {axis=}. Pass one axis to"
            " compute a vector-norm, or two axes to compute a matrix-norm."
        )


def vector_norm(
    x: TensorLike, /, *,
    axis: Axis = None, keepdims: bool = False,
    ord: int | str = 2
) -> core.BaseTensor:
    x, = dannet.core._args_to_tensor('linalg.vector_norm', x)
    if ord is None or ord == 2:
        return dannet.sqrt(
            dannet.sum(dannet.real(x * x.conj()),
                       axis=axis,
                       keepdims=keepdims
                       )
        )
    elif ord == np.inf:
        return dannet.max(
            dannet.abs(x),
            axis=axis, keepdims=keepdims, initial=0
        )
    elif ord == -np.inf:
        return dannet.min(dannet.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
        return dannet.sum(
            x != 0,
            dtype=dtypes.finfo(x.dtype).dtype,
            axis=axis, keepdims=keepdims
        )
    elif ord == 1:
        return dannet.sum(dannet.abs(x), axis=axis, keepdims=keepdims)
    elif isinstance(ord, str):
        msg = f"Invalid order '{ord}' for vector norm."
        if ord == "inf":
            msg += "Use 'jax.numpy.inf' instead."
        if ord == "-inf":
            msg += "Use '-jax.numpy.inf' instead."
        raise ValueError(msg)
    else:
        abs_x = dannet.abs(x)
        ord_arr = dannet.array(ord).astype(abs_x.dtype)
        ord_inv = dannet.array(1 / ord).astype(abs_x.dtype)
        raise NotImplementedError("power not implemented")
        out = dannet.sum(abs_x ** ord_arr, axis=axis, keepdims=keepdims)
        return dannet.power(out, ord_inv)


def matrix_norm(
    x: TensorLike, /, *,
    keepdims: bool = False, ord: str | int = 'fro'
) -> core.BaseTensor:
    x, = dannet.core._args_to_tensor('linalg.matrix_norm', x)
    return norm(x, ord=ord, keepdims=keepdims, axis=(-2, -1))


def outer(x1: TensorLike, x2: TensorLike, /) -> core.BaseTensor:
    x1, x2 = dannet.core._args_to_tensor("linalg.outer", x1, x2)
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError(
            "Input arrays must be one-dimensional, "
            f"but they are {x1.ndim=} {x2.ndim=}"
        )
    return x1[:, None] * x2[None, :]
