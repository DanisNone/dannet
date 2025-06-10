import math
import operator
from typing import Any, Callable, NoReturn, Sequence, SupportsIndex
import dannet as dt
from dannet import dtypes
from dannet.core import Tensor, TensorInfo
from dannet.gradient import GradientOp

from dannet.compiler.impl.unary import (
    copy, astype
)

def broadcast_shapes(*shapes: dt.typing.ShapeLike) -> tuple[int, ...]:
    norm_shapes = tuple(dt.utils.normalize_shape(s) for s in shapes)
    ndim = max(map(len, norm_shapes))

    result = [1] * ndim
    for shape in norm_shapes:
        shape = (1, ) * (ndim - len(shape)) + shape
        for i, (dim1, dim2) in enumerate(zip(result, shape)):
            if dim1 == dim2:
                continue
            elif dim1 == 1:
                result[i] = dim2
            else:
                raise ValueError(
                    f"fail broadcast shapes: {shapes=}"
                )
    return tuple(result)


def _not_implemented_grad(name: str) -> Callable[..., NoReturn]:
    def _(*args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(f"gradient for {name} not implemented")
    return _


def _broadcast_to(
    x: Tensor,
    shape: dt.typing.ShapeLike
) -> Tensor:
    shape = dt.utils.normalize_shape(shape)
    if broadcast_shapes(x.shape, shape) != shape:
        raise ValueError(
            f"fail broadcast shape {x.shape} to {shape}"
        )

    pad_shape = (1, ) * (len(shape) - x.ndim) + x.shape
    new_strides = [0] * (len(shape) - x.ndim) + list(x.strides)

    for i in range(len(shape)):
        if pad_shape[i] != shape[i]:
            new_strides[i] = 0

    return Tensor(
        x._buffer,
        TensorInfo(shape, x.dtype, tuple(new_strides))
    )


broadcast_to = GradientOp(
    _broadcast_to,
    _not_implemented_grad('broadcast_to'),
    nondiff_argnum=(1,)
)

def _transpose_op(
    x: Tensor,
    axes: dt.typing.ShapeLike | None = None
) -> Tensor:
    if axes is None:
        axes = range(x.ndim)[::-1]
    axes = dt.utils.normalize_axis_tuple(axes, x.ndim, "x")
    if sorted(axes) != list(range(x.ndim)):
        # TODO: add message
        raise ValueError

    new_shape = tuple(x.shape[i] for i in axes)
    new_strides = tuple(x.strides[i] for i in axes)
    return Tensor(
        x._buffer,
        TensorInfo(new_shape, x.dtype, new_strides)
    )

def _transpose_grad(
    grad: Tensor, out: Tensor,
    args: tuple[Tensor, dt.typing.ShapeLike], kwargs: Any
):
    if len(args) == 1:
        x, axes = args[0], None
    else:
        x, axes = args
    if axes is None:
        axes = range(x.ndim)[::-1]
    axes = dt.utils.normalize_axis_tuple(axes, x.ndim, "x")
    inv_axes = [axes.index(i) for i in range(x.ndim)]
    return dt.transpose(grad, inv_axes)
    

transpose = GradientOp(
    _transpose_op,
    _transpose_grad,
    nondiff_argnum=(1,)
)

def _reshape_op(
    x: Tensor,
    shape: dt.typing.ShapeLike
) -> Tensor:
    shape_ = list(dt.utils.normalize_shape(shape))
    
    if min(shape_, default=0) < -1:
        raise ValueError(f'Invalid shape: {shape_}')
    if shape_.count(-1) > 1:
        raise ValueError('Only one dimension can be set to -1 in reshape.')

    if shape_.count(-1) == 1:
        i = shape_.index(-1)
        s1 = x.size
        s2 = math.prod(shape_[:i] + shape_[i + 1:])
        if s2 == 0:
            raise ValueError(
                'Reshape target shape contains zero-length dimensions.'
            )
        if s1 % s2 != 0:
            raise ValueError(
                f'Cannot infer dimension size for -1: '
                f'{s1} is not divisible by {s2}.'
            )
        shape_[i] = s1 // s2

    if x.size != math.prod(shape_):
        raise ValueError(
            f'Total size of new shape {shape_} '
            f'must be unchanged from input shape {x.shape}.'
        )    

    # TODO: implement reshape without copy
    x = dt.copy(x)
    return Tensor(
        x._buffer,
        TensorInfo(shape_, x.dtype)
    )

def _reshape_grad(
    grad: Tensor, out: Tensor,
    args: tuple[Tensor, dt.typing.ShapeLike], kwargs: Any
):
    return dt.reshape(grad, args[0].shape)
    

reshape = GradientOp(
    _reshape_op,
    _reshape_grad,
    nondiff_argnum=(1,)
)


def full(
    shape: dt.typing.ShapeLike,
    fill_value: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None, *,
    device: dt.Device | None = None
) -> Tensor:
    res = dt.array(fill_value, dtype=dtype, device=device)
    return dt.broadcast_to(res, shape)


def full_like(
    a: dt.typing.TensorLike,
    fill_value: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> Tensor:
    a = dt.array(a, device=device)
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return full(shape, fill_value, dtype, device=device)


def zeros(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike = dtypes.float64,
    *, device: dt.Device | None = None
) -> Tensor:
    return full(shape, 0, dtype, device=device)


def zeros_like(
    a: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> Tensor:
    return full_like(a, fill_value=0, dtype=dtype, shape=shape, device=device)


def ones(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike = dtypes.float64,
    *, device: dt.Device | None = None
) -> Tensor:
    return full(shape, 1, dtype, device=device)


def ones_like(
    a: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> Tensor:
    return full_like(a, fill_value=1, dtype=dtype, shape=shape, device=device)


def squeeze(
    x: dt.typing.TensorLike,
    axis: dt.typing.ShapeLike | None = None
):
    x = dt.array(x)

    if axis is None:
        axis = [i for i, dim in enumerate(x.shape) if dim == 1]
    axis = dt.utils.normalize_axis_tuple(axis, x.ndim, "x")

    for a in axis:
        if x.shape[a] != 1:
            raise ValueError(
                f'cannot select an axis to squeeze '
                f'out which has size not equal to one, '
                f'axis {a} has size {x.shape[a]}'
            )

    new_shape = [dim for i, dim in enumerate(x.shape) if i not in axis]
    return reshape(x, new_shape)

__all__ = [
    "copy", "astype",
    "broadcast_shapes", "broadcast_to",
    "transpose", "reshape", "squeeze",
    "full", "zeros", "ones",
    "full_like", "zeros_like", "ones_like",

    "eye"
]
