import math
import operator
from typing import Hashable, Sequence, SupportsIndex
import dannet
from dannet.lib import core, dtypes, utils
from dannet.lib.core import SymbolicTensor, normalize_slices
from dannet.lib.typing import ShapeLike


class AsStrides(SymbolicTensor):
    def __init__(
        self,
        x: SymbolicTensor,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int = 0,
        dtype: dtypes.DannetDtype | None = None
    ):
        self.x = x
        self._shape = shape
        self._strides = strides
        self._offset = offset
        self._buffer = x.buffer
        if dtype is None:
            dtype = x.dtype
        self._dtype = dtype

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x]

    def get_config(self) -> dict[str, Hashable]:
        return {}


def broadcast_to(x: core.BaseTensor, shape: tuple[int, ...]) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if len(shape) < x.ndim:
        raise ValueError(
            "Cannot broadcast to shape with fewer dimensions: "
            f"arr_shape={x.shape} shape={shape}"
        )

    out_shape = core._broadcast_shapes_with_name("broadcast", x.shape, shape)
    if shape != out_shape:
        raise ValueError(
            f"Shape {shape} is not broadcastable from shape {x.shape}"
        )

    strides = [0] * (len(shape) - len(x.shape))
    for s, d in zip(x.strides, shape[len(strides):]):
        strides.append(0 if d == 1 else s)

    return AsStrides(
        x=x,
        shape=shape,
        strides=tuple(strides),
        offset=x.buffer_offset
    )


def flip(x: core.BaseTensor, axes: core.Axes = None) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if axes is None:
        axes = list(range(x.ndim))
    norm_axes = utils.normalize_axis_tuple(axes, x.ndim, "x")

    norm_axes = tuple(sorted(norm_axes))
    orig_strides = x._strides

    new_strides = list(orig_strides)
    for ax in norm_axes:
        new_strides[ax] = -orig_strides[ax]

    offset = x.buffer_offset
    for ax in norm_axes:
        offset += (x.shape[ax] - 1) * orig_strides[ax]

    return AsStrides(
        x, x.shape,
        tuple(new_strides), offset
    )


def transpose(x: core.BaseTensor, axes: core.Axes = None) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if axes is None:
        axes = tuple(reversed(range(x.ndim)))

    norm_axes = utils.normalize_axis_tuple(axes, x.ndim, "transpose")
    if len(norm_axes) != x.ndim:
        raise ValueError("Axes must match tensor dimensions")

    new_shape = tuple(x.shape[ax] for ax in norm_axes)
    new_strides = tuple(x.strides[ax] for ax in norm_axes)

    return AsStrides(
        x=x,
        shape=new_shape,
        strides=new_strides,
        offset=x.buffer_offset
    )


def squeeze(x: core.BaseTensor, axis: core.Axes = None) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if axis is None:
        axis = tuple(i for i, s in enumerate(x.shape) if s == 1)
    norm_axis = utils.normalize_axis_tuple(axis, x.ndim, "axis")

    if any(x.shape[i] != 1 for i in norm_axis):
        raise ValueError
    new_shape = [s for i, s in enumerate(x.shape) if i not in norm_axis]
    new_strides = [s for i, s in enumerate(x.strides) if i not in norm_axis]

    return AsStrides(
        x=x,
        shape=tuple(new_shape),
        strides=tuple(new_strides),
        offset=x.buffer_offset
    )


def expand_dims(x: core.BaseTensor, axis: ShapeLike) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if not isinstance(axis, Sequence):
        axis = [axis]
    norm_axis = utils.normalize_axis_tuple(axis, x.ndim + len(axis), "axis")
    new_shape = list(x.shape)
    new_strides = list(x.strides)

    for ax in sorted(norm_axis):
        new_shape.insert(ax, 1)
        new_strides.insert(ax, 0)

    return AsStrides(
        x=x,
        shape=tuple(new_shape),
        strides=tuple(new_strides),
        offset=x.buffer_offset
    )


py_slice = slice


def slice(
    x: core.BaseTensor,
    slices: tuple[SupportsIndex | py_slice] | list[SupportsIndex | py_slice]
) -> core.BaseTensor:
    x = core.to_symbolic(x)

    if not isinstance(slices, (list, tuple)):
        raise TypeError('slices must be a sequence of tuples')
    if len(slices) > x.ndim:
        raise ValueError(
            f'Too many slice tuples ({len(slices)}) '
            f'for tensor of ndim {x.ndim}'
        )

    norm_slices: list[tuple[int | None, int | None, int | None]] = []
    for i, el in enumerate(slices):
        if isinstance(el, SupportsIndex):
            s = operator.index(el)
            norm_slices.append((s, s+1, 1))
        elif isinstance(el, py_slice):
            norm_slices.append((el.start, el.stop, el.step))

    norm_slices = [
        *norm_slices,
        *[(None, None, None)] * (x.ndim - len(slices))
    ]

    new_slices, new_shape, new_strides = normalize_slices(
        norm_slices, x.shape, x.strides)
    offset = x.buffer_offset

    for stride, (start, _, _) in zip(new_strides, new_slices):
        offset += stride * start

    return AsStrides(
        x,
        tuple(new_shape),
        tuple(new_strides),
        offset
    )


def real(x: core.BaseTensor) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if not dtypes.is_complex_dtype(x.dtype):
        return x

    dtype = dtypes.real_part_of_complex(x.dtype)
    strides = tuple(s * 2 for s in x.strides)
    offset = x.buffer_offset * 2
    return AsStrides(
        x,
        x.shape,
        strides,
        offset,
        dtype
    )


def imag(x: core.BaseTensor) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if not dtypes.is_complex_dtype(x.dtype):
        return dannet.zeros_like(x)

    dtype = dtypes.real_part_of_complex(x.dtype)
    strides = tuple(s * 2 for s in x.strides)
    offset = x.buffer_offset * 2 + 1
    return AsStrides(
        x,
        x.shape,
        strides,
        offset,
        dtype
    )


def ravel(x: core.BaseTensor) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if x.strides != core.default_strides(x.shape):
        raise ValueError("x must be continuous")

    return AsStrides(
        x,
        (x.size, ),
        (1, ),
        x._offset
    )


def reshape(x: core.BaseTensor, shape: tuple[int, ...]) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if x.strides != core.default_strides(x.shape):
        raise ValueError("x must be continuous")
    if min(shape, default=0) < -1:
        raise ValueError(f'Invalid shape: {shape}')
    if 0 in shape:
        raise NotImplementedError

    if shape.count(-1) > 1:
        raise ValueError("Only one dimension can be set to -1 in reshape.")
    if shape.count(-1) == 1:
        size = -math.prod(shape)
        dim, rem = divmod(x.size, size)
        if rem != 0:
            raise ValueError(
                f'Cannot infer dimension size for -1: '
                f'{x.size} is not divisible by {size}.'
            )
        shape = tuple(dim if s == -1 else s for s in shape)
    if x.size != math.prod(shape):
        raise ValueError(
            f'Total size of new shape {shape} '
            f'must be unchanged from input shape {x.shape}.'
        )

    return AsStrides(
        x,
        shape,
        core.default_strides(shape),
        x._offset
    )
