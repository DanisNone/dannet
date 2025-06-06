from __future__ import annotations
from typing import SupportsIndex

import numpy as np
import dannet as dt


def normalize_shape(
    shape: dt.typing.ShapeLike
) -> tuple[int, ...]:
    if isinstance(shape, SupportsIndex):
        shape = [shape]
    shape_norm = tuple(int(dim) for dim in shape)
    if min(shape_norm, default=1) <= 0:
        raise ValueError('All dims of shape must be greater than 0')
    return shape_norm


def convert_to_tensor(x: dt.typing.TensorLike) -> dt.core.TensorBase:
    if isinstance(x, dt.core.TensorBase):
        return x

    if isinstance(x, (np.generic, list, tuple)):
        return dt.constant(x)
    if isinstance(x, bool):
        return dt.constant(x, dt.dtype.py_bool)
    if isinstance(x, int):
        return dt.constant(x, dt.dtype.py_int)
    if isinstance(x, float):
        return dt.constant(x, dt.dtype.py_float)
    if isinstance(x, complex):
        return dt.constant(x, dt.dtype.py_complex)
    if hasattr(x, '__array__'):
        x = np.asarray(x)
        return dt.constant(x)

    raise TypeError(f'Fail convert to Tensor: {x!r}')


def broadcast_shapes(*shapes: dt.typing.ShapeLike) -> tuple[int, ...]:
    norm_shapes = tuple(normalize_shape(shape) for shape in shapes)
    ndim = max(map(len, norm_shapes), default=0)

    result = [1] * ndim
    for shape in norm_shapes:
        shape = (1, ) * (ndim - len(norm_shapes)) + shape
        for i, (dim1, dim2) in enumerate(zip(result, shape)):
            if dim1 == dim2 or dim2 == 1:
                continue
            elif dim1 == 1:
                result[i] = dim2
            else:
                raise ValueError(f'Cannot broadcast shapes: {shapes}')
    return tuple(result)


def broadcast_shape_to(
    shape1: dt.typing.ShapeLike,
    shape2: dt.typing.ShapeLike
) -> tuple[int, ...]:
    shape = broadcast_shapes(shape1, shape2)
    if shape != shape2:
        raise ValueError(f'Fail broadcast {shape1} to {shape2}')
    return shape


def normalize_axis_tuple(
    axis: dt.typing.Axis | None,
    x: dt.core.TensorBase | int,
) -> tuple[int, ...]:
    if isinstance(x, int):
        ndim = x
    else:
        ndim = x.ndim

    if axis is None:
        axis = range(ndim)
    if isinstance(axis, SupportsIndex):
        axis = [axis]

    res_axis = [normalize_axis_index(i, ndim) for i in axis]
    if len(set(res_axis)) != len(res_axis):
        raise ValueError(f'repeated axis: {axis}')

    return tuple(res_axis)


def normalize_axis_index(
    axis: SupportsIndex,
    ndim: int,
    msg_prefix: str | None = None
):
    axis = int(axis)
    if not (-ndim <= axis < ndim):
        msg = (
            f'axis {axis} is out of bounds '
            f'for tensor of dimension {ndim}'
        )
        if msg_prefix is not None:
            msg = f'{msg_prefix}: {msg}'
        raise ValueError(msg)
    if axis < 0:
        axis += ndim
    return axis
