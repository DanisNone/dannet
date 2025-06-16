import operator
from typing import SupportsIndex
import dannet as dt
import numpy as np

from dannet import lib
from dannet.lib.core import TensorLike
from dannet.lib.dtypes import DTypeLike
from dannet.lib.typing import ShapeLike


def empty(
    shape: ShapeLike, dtype: DTypeLike = float,
    *, device: dt.Device | None = None
) -> lib.core.BaseTensor:
    shape_ = lib.utils.normalize_shape(shape)
    dtype_ = lib.utils.normalize_dtype(dtype)
    itemsize = lib.dtypes.itemsize(dtype_)

    size = 1
    for dim in shape_:
        size *= dim

    if device is None:
        device = dt.current_device()
    buffer = lib.core.ConcreteBuffer(device, size * itemsize)
    return lib.core.ConcreteTensor(
        shape_,
        lib.core.default_strides(shape_),
        0,
        dtype_,
        buffer,
        event=None
    )


def zeros(
    shape: ShapeLike, dtype: DTypeLike = float, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    return dt.broadcast_to(dt.array(0, dtype, device=device), shape)


def ones(
    shape: ShapeLike, dtype: DTypeLike = float, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    return dt.broadcast_to(dt.array(1, dtype, device=device), shape)


def full(
    shape: ShapeLike,
    fill_value: TensorLike,
    dtype: DTypeLike | None = None, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    fill_value = dt.array(fill_value, dtype=dtype, device=device)
    return dt.broadcast_to(fill_value, shape).astype(dtype)


def empty_like(
    a: TensorLike,
    dtype: DTypeLike | None = None,
    shape: ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    a = dt.array(a, device=device)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    return empty(shape, dtype, device=device)


def zeros_like(
    a: TensorLike,
    dtype: DTypeLike | None = None,
    shape: ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    a = dt.array(a, device=device)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    return zeros(shape, dtype, device=device)


def ones_like(
    a: TensorLike,
    dtype: DTypeLike | None = None,
    shape: ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    a = dt.array(a, device=device)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    return ones(shape, dtype, device=device)


def full_like(
    a: TensorLike,
    fill_value: TensorLike,
    dtype: DTypeLike | None = None,
    shape: ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    a = dt.array(a, device=device)
    if shape is None:
        shape = a.shape
    if dtype is None:
        dtype = a.dtype
    return full(shape, fill_value, dtype, device=device)


def eye(
    N: SupportsIndex,
    M: SupportsIndex | None = None,
    k: SupportsIndex = 0,
    dtype: DTypeLike = float,
    device: dt.Device | None = None
) -> lib.core.BaseTensor:
    # TODO: implement without numpy
    N = operator.index(N)
    if M is not None:
        M = operator.index(M)
    k = operator.index(k)
    return dt.array(np.eye(N, M, k, dtype), device=device)
