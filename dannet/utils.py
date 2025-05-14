from __future__ import annotations
from typing import SupportsIndex

import numpy as np
import dannet as dt


def normalize_shape(shape: dt.typing.ShapeLike | SupportsIndex) -> tuple[int, ...]:
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
    if isinstance(x, int):
        return dt.constant(x, dt.dtype.int_dtype)
    if isinstance(x, float):
        return dt.constant(x, dt.dtype.float_dtype)    
    if hasattr(x, '__array__'):
        x = np.asarray(x)
        return dt.constant(x)
    

    raise TypeError(f'Fail convert to Tensor: {x!r}')

def broadcast_shapes(*shapes: dt.typing.ShapeLike) -> tuple[int, ...]:
    shapes = tuple(normalize_shape(shape) for shape in shapes)
    ndim = max(map(len, shapes), default=0)

    result = [1] * ndim
    for shape in shapes:
        shape = (1, ) * (ndim - len(shape)) + shape
        for i, (dim1, dim2) in enumerate(zip(result, shape)):
            if dim1 == dim2 or dim2 == 1:
                continue
            elif dim1 == 1:
                result[i] = dim2
            else:
                raise ValueError(f'Cannot broadcast shapes: {shapes}')
    return tuple(result)

def broadcast_shape_to(shape1: dt.typing.ShapeLike, shape2: dt.typing.ShapeLike) -> tuple[int, ...]:
    shape = broadcast_shapes(shape1, shape2)
    if shape != shape2:
        raise ValueError(f'Fail broadcast {shape1} to {shape2}')
    return shape