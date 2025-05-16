import math

import numpy as np
import dannet as dt


class _BroadcastTo(dt.core.TensorBase):
    def __init__(self, x, new_shape):
        self.x = dt.convert_to_tensor(x)

        self._shape = dt.utils.normalize_shape(new_shape)
        self._dtype = self.x._dtype

        pad = len(self._shape) - len(self.x._shape)
        if pad < 0:
            raise ValueError(f'fail broadcast {self.x} to {self._shape}')

        shape = [1] * pad + list(self.x._shape)

        strides = [0] * pad + list(self.x._strides)

        for i in range(len(shape)):
            if shape[i] == self._shape[i]:
                continue
            elif shape[i] == 1:
                strides[i] = 0
            else:
                raise ValueError(f'fail broadcast {self.x} to {self._shape}')

        self._strides = tuple(strides)
        self._buffer = self.x._buffer
        self._buffer_offset = self.x._buffer_offset

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [dt.reduce_to(grad, self.x._shape)]

    def get_config(self):
        return {}


class _Cast(dt.core.TensorBase):
    def __init__(self, x, new_dtype):
        self.x = dt.convert_to_tensor(x)

        self._shape = self.x._shape
        self._dtype = dt.dtype.normalize_dtype(new_dtype)

        self._strides = self._default_strides()
        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [grad]

    def get_config(self):
        return {}


class _Reshape(dt.core.TensorBase):
    def __init__(self, x, new_shape):
        self.x = dt.convert_to_tensor(x)

        new_shape = [int(dim) for dim in new_shape]
        if min(new_shape, default=0) < -1:
            raise ValueError(f'Invalid shape: {new_shape}')
        if new_shape.count(-1) > 1:
            raise ValueError('Only one dimension can be set to -1 in reshape.')

        if new_shape.count(-1) == 1:
            i = new_shape.index(-1)
            s1 = self.x.size
            s2 = math.prod(new_shape[:i] + new_shape[i + 1:])
            if s2 == 0:
                raise ValueError(
                    'Reshape target shape contains zero-length dimensions.'
                )
            if s1 % s2 != 0:
                raise ValueError(
                    f'Cannot infer dimension size for -1: '
                    f'{s1} is not divisible by {s2}.'
                )
            new_shape[i] = s1 // s2

        if self.x.size != math.prod(new_shape):
            raise ValueError(
                f'Total size of new shape {new_shape} '
                f'must be unchanged from input shape {self.x.shape}.'
            )

        self._shape = tuple(new_shape)
        self._dtype = self.x.dtype

        # TODO: implement smart reshape
        if self.x._is_default_strides():
            self._strides = self._default_strides()
            self._buffer = self.x._buffer
            self._buffer_offset = self.x._buffer_offset
        else:
            self._strides = self._default_strides()
            self._buffer = dt.core.TensorBuffer(self)
            self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [reshape(grad, self.x.shape)]

    def get_config(self):
        return {}


class _Transpose(dt.core.TensorBase):
    def __init__(self, x, axes=None):
        self.x = dt.convert_to_tensor(x)

        if axes is None:
            axes = range(self.x.ndim)[::-1]
        axes = [a if a >= 0 else a + self.x.ndim for a in axes]

        if sorted(axes) != list(range(self.x.ndim)):
            raise ValueError(f'Invalid axes for transpose: {axes}')

        self._shape = tuple(self.x._shape[a] for a in axes)
        self._dtype = self.x._dtype

        self._strides = tuple(self.x._strides[a] for a in axes)
        self._buffer = self.x._buffer
        self._buffer_offset = self.x._buffer_offset

        self._axes = tuple(axes)

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        inv = [self._axes.index(i) for i in range(self.x.ndim)]
        return [transpose(grad, inv)]

    def get_config(self):
        return {'axes': self._axes}


class _Flip(dt.core.TensorBase):
    def __init__(self, x, axes):
        self.x = dt.convert_to_tensor(x)
        ndim = self.x.ndim

        if axes is None:
            axes = list(range(ndim))
        elif hasattr(axes, '__index__'):
            axes = (int(axes), )
        axes = tuple(axes)

        norm_axes = []
        for a in axes:
            if a < 0:
                a += ndim
            if not (0 <= a < ndim):
                raise ValueError(
                    f'Invalid axis {a} for Flip with tensor of ndim {ndim}')
            norm_axes.append(a)

        self._axes = tuple(sorted(set(norm_axes)))

        self._shape = self.x._shape
        self._dtype = self.x._dtype

        orig_strides = self.x._strides

        new_strides = list(orig_strides)
        for ax in self._axes:
            new_strides[ax] = -orig_strides[ax]

        offset = self.x._buffer_offset
        for ax in self._axes:
            offset += (self._shape[ax] - 1) * orig_strides[ax]

        self._strides = tuple(new_strides)
        self._buffer = self.x._buffer
        self._buffer_offset = offset

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [flip(grad, self._axes)]

    def get_config(self):
        return {'axes': self._axes}


class _Copy(dt.core.TensorBase):
    def __init__(self, x):
        self.x = dt.convert_to_tensor(x)

        self._shape = self.x._shape
        self._dtype = self.x._dtype

        self._strides = self._default_strides()
        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [grad]

    def get_config(self):
        return {}


class _Pad(dt.core.TensorBase):
    def __init__(self, x, paddings):
        self.x = dt.convert_to_tensor(x)
        norm_paddings: list[tuple[int, int]] = []
        for p in paddings:
            if hasattr(p, '__index__'):
                p1, p2 = p, p
            else:
                p1, p2 = p

            p1, p2 = int(p1), int(p2)
            if p1 < 0 or p2 < 0:
                raise ValueError(
                    f'Invalid padding {(p1, p2)}, must be non-negative ints'
                )

            norm_paddings.append((p1, p2))

        norm_paddings += ((0, 0), ) * (self.x.ndim - len(norm_paddings))
        self._paddings = norm_paddings

        if len(self._paddings) != self.x.ndim:
            raise ValueError(
                f'Paddings length {len(paddings)} '
                f'must match tensor ndim {self.x.ndim}'
            )

        for p in self._paddings:
            if len(p) != 2:
                raise ValueError(f'Invalid padding: {p}')
            p1, p2 = map(int, p)
            if p1 >= 0 and p2 >= 0:
                raise ValueError(
                    f'Invalid padding {p}, must be non-negative ints')

        self._shape = tuple(
            self.x.shape[i] + paddings[i][0] + paddings[i][1]
            for i in range(self.x.ndim)
        )

        self._dtype = self.x.dtype

        self._strides = self._default_strides()
        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        slices = []
        for i, (before, _after) in enumerate(self._paddings):
            start = before
            stop = before + self.x.shape[i]
            slices.append((start, stop, None))
        return [slice(grad, tuple(slices))]

    def get_config(self):
        return {'paddings': self._paddings}


class _Slice(dt.core.TensorBase):
    def __init__(self, x, slices):
        self.x = dt.convert_to_tensor(x)
        ndim = self.x.ndim

        if not isinstance(slices, (list, tuple)):
            raise TypeError('slices must be a sequence of tuples')
        if len(slices) > ndim:
            raise ValueError(
                f'Too many slice tuples ({len(slices)}) '
                f'for tensor of ndim {ndim}'
            )
        slices = list(slices) + [(None, None, None)] * (ndim - len(slices))

        orig_shape = self.x.shape
        orig_strides = self.x._strides
        new_shape = []
        new_strides = []
        offset = self.x._buffer_offset

        for i, (start, stop, step) in enumerate(slices):
            dim = orig_shape[i]
            stride = orig_strides[i]

            if step is None:
                step = 1
            if step == 0:
                raise ValueError('slice step cannot be zero')
            if step < 0:
                default_start = dim - 1
                default_stop = -1
            else:
                default_start = 0
                default_stop = dim

            if start is None:
                start = default_start
            elif start < 0:
                start += dim
            if stop is None:
                stop = default_stop
            elif stop < 0:
                stop += dim

            if step < 0:
                start = max(0, min(start, dim - 1))
                stop = max(-1, min(stop, dim - 1))
                length = max(0, math.ceil((stop - start) / step))
            else:
                start = max(0, min(start, dim))
                stop = max(0, min(stop, dim))
                length = max(0, math.ceil((start - stop) / (-step)))
            new_shape.append(length)
            new_strides.append(stride * step)

            offset += stride * start

        if 0 in new_shape:
            raise NotImplementedError(
                'TensorBase not support slice with zero size')

        self._shape = tuple(new_shape)
        self._dtype = self.x.dtype
        self._strides = tuple(new_strides)
        self._buffer = self.x._buffer
        self._buffer_offset = offset

        self._slices = tuple(slices)

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        pads = []
        for (start, stop, step), out_dim in zip(self._slices, grad.shape):
            if step not in (None, 1):
                raise NotImplementedError(
                    'Gradient for step != 1 slicing is not yet supported')

            dim = self.x.shape[len(pads)]
            start = 0 if start is None else (
                start + dim if start < 0 else start)
            pads.append((start, dim - start - out_dim))
        zero = zeros(self.x.shape, self.x.dtype)
        return [pad(grad, pads) + zero]

    def get_config(self):
        return {'slices': self._slices}


class _Gather(dt.core.TensorBase):
    def __init__(self, x, indices):
        self.x = dt.convert_to_tensor(x)
        self.indices = dt.cast(indices, dt.dtype.int_dtype)

        self._shape = self.indices.shape + self.x.shape[1:]
        self._dtype = self.x.dtype

        self._strides = self._default_strides()
        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x, self.indices]

    def compute_gradients(self, grad):
        flat_dim = math.prod(self.x.shape[1:])

        one_hot = dt.one_hot(self.indices, self.x.shape[0], dtype=grad.dtype)

        grad_2d = dt.reshape(grad, (-1, flat_dim))
        one_hot_2d = dt.reshape(one_hot, (-1, self.x.shape[0]))

        grad_x_flat = dt.matmul(one_hot_2d, grad_2d, transpose_a=True)

        grad_x = dt.reshape(grad_x_flat, self.x.shape)
        return [grad_x, dt.zeros_like(self.indices)]

    def get_config(self):
        return {}


class _OneHot(dt.core.TensorBase):
    def __init__(self, indices, depth, dtype):
        self.indices = dt.cast(indices, dt.dtype.int_dtype)
        self._depth = int(depth)

        if self._depth <= 0:
            raise ValueError(
                f'depth must be positivev integer, not {self._depth}')

        self._shape = (*self.indices._shape, depth)
        self._dtype = dt.dtype.normalize_dtype(dtype)

        self._strides = self._default_strides()
        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.indices]

    def compute_gradients(self, grad):
        return [dt.zeros_like(grad)]

    def get_config(self):
        return {'depth': self._depth}


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype
    return broadcast_to(cast(0, dtype), shape)


def ones(shape, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype
    return broadcast_to(cast(1, dtype), shape)


def zeros_like(x, dtype: dt.typing.DTypeLike | None = None):
    x = dt.convert_to_tensor(x)
    if dtype is None:
        dtype = x.dtype
    return broadcast_to(cast(0, dtype), x.shape)


def ones_like(x, dtype: dt.typing.DTypeLike | None = None):
    x = dt.convert_to_tensor(x)
    if dtype is None:
        dtype = x.dtype
    return broadcast_to(cast(1, dtype), x.shape)


def broadcast_to(x, shape):
    x = dt.convert_to_tensor(x)
    y = _BroadcastTo(x, shape)

    if x.shape == y.shape:
        y = x
    return dt.core._node_prepare(y)


def reduce_to(x, shape):
    x = dt.convert_to_tensor(x)
    shape = dt.utils.normalize_shape(shape)

    if x.ndim < len(shape):
        raise ValueError(f'Fail reduce {x} to {shape}')

    pad_shape = (1, ) * (x.ndim - len(shape)) + shape
    sum_axis = []

    for i, (dim1, dim2) in enumerate(zip(x._shape, pad_shape)):
        if dim1 == dim2:
            continue
        elif dim2 == 1:
            sum_axis.append(i)
        else:
            raise ValueError(f'Fail reduce {x} to {shape}')
    return dt.reshape(dt.sum(x, axis=sum_axis), shape)


def cast(x: dt.typing.TensorLike, dtype: dt.typing.DTypeLike | None):
    x = dt.convert_to_tensor(x)
    if dtype is None:
        return x
    y = _Cast(x, dtype)
    if x.dtype == y.dtype:
        y = x
    return dt.core._node_prepare(y)


def reshape(x, shape):
    x = dt.convert_to_tensor(x)
    y = _Reshape(x, shape)

    if x.shape == y.shape:
        y = x
    return dt.core._node_prepare(y)


def squeeze(x, axis=None):
    x = dt.convert_to_tensor(x)
    shape = x.shape
    ndim = x.ndim

    if axis is None:
        axes = [i for i, dim in enumerate(shape) if dim == 1]
    else:
        if isinstance(axis, (list, tuple)):
            axes = list(axis)
        else:
            axes = [axis]
        axes = [a + ndim if a < 0 else a for a in axes]

    for a in axes:
        if a < 0 or a >= ndim:
            raise ValueError(
                f'axis {a} is out of bounds for tensor of dimension {ndim}')
        if shape[a] != 1:
            raise ValueError(
                f'cannot select an axis to squeeze '
                f'out which has size not equal to one, '
                f'axis {a} has size {shape[a]}'
            )

    new_shape = [dim for i, dim in enumerate(shape) if i not in axes]
    return reshape(x, new_shape)


def expand_dims(x, axis):
    x = dt.convert_to_tensor(x)

    if hasattr(axis, '__index__'):
        axis = (int(axis), )
    axis = tuple(axis)

    if len(set(axis)) != len(axis):
        raise ValueError(f'Duplicate axes: {axis}')

    normalized_axes = []
    for ax in axis:
        if ax < 0:
            ax = x.ndim + 1 + ax
        if ax < 0 or ax > x.ndim:
            raise ValueError(
                f'Axis {ax} out of bounds for tensor of dimension {x.ndim}')
        normalized_axes.append(ax)

    shape = list(x.shape)
    for ax in sorted(normalized_axes, reverse=True):
        shape.insert(ax, 1)

    return reshape(x, shape)


def transpose(x, axes=None):
    x = dt.convert_to_tensor(x)
    y = _Transpose(x, axes)

    if list(y._axes) == sorted(y._axes):
        y = x
    return dt.core._node_prepare(y)


def swapaxes(x, axis1, axis2):
    x = dt.convert_to_tensor(x)

    axes = list(range(x.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]

    return dt.transpose(x, axes)


def moveaxis(a, source, destination):
    source = dt.utils.normalize_axis_tuple(source, a.ndim)
    destination = dt.utils.normalize_axis_tuple(destination, a.ndim)

    if len(source) != len(destination):
        raise ValueError(
            '`source` and `destination` must have the same number of elements')

    order = list(range(a.ndim))
    for s, d in sorted(zip(source, destination), key=lambda x: x[1]):
        order.pop(s)
        order.insert(d, s)
    return dt.transpose(a, order)


def flip(x, axis=None):
    x = dt.convert_to_tensor(x)
    y = _Flip(x, axis)
    return dt.core._node_prepare(y)


def pad(x, paddings):
    x = dt.convert_to_tensor(x)
    y = _Pad(x, paddings)
    if x.shape == y.shape:
        y = x
    return dt.core._node_prepare(y)


def copy(x):
    x = dt.convert_to_tensor(x)
    y = _Copy(x)
    return dt.core._node_prepare(y)


def slice(x, slices):
    y = _Slice(x, slices)
    return dt.core._node_prepare(y)


def take(x, indices, axis=None):
    x = dt.convert_to_tensor(x)
    indices = dt.cast(indices, dt.dtype.int_dtype)

    if axis is None:
        flat = dt.reshape(x, (-1,))
        res = _Gather(flat, indices)
        return dt.core._node_prepare(res)

    ndim = x.ndim
    norm_axis = axis if axis >= 0 else axis + ndim
    if not (0 <= norm_axis < ndim):
        raise ValueError(f'axis {axis} out of range for tensor of ndim {ndim}')

    if norm_axis != 0:
        axes = [norm_axis] + [i for i in range(ndim) if i != norm_axis]
        x = dt.transpose(x, axes)

    head = x.shape[0]
    tail = math.prod(x.shape[1:])
    x_flat = dt.reshape(x, (head, tail))

    gathered = _Gather(x_flat, indices)
    gathered = dt.core._node_prepare(gathered)

    out_shape = indices.shape + x.shape[1:]
    out = dt.reshape(gathered, out_shape)

    if norm_axis != 0:
        i_ndim = indices.ndim

        perm2 = [
            *range(i_ndim, norm_axis + i_ndim),
            *range(i_ndim),
            *range(norm_axis + i_ndim, ndim + i_ndim - 1)
        ]
        out = dt.transpose(out, perm2)

    return out


def one_hot(x, depth, axis=-1, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype
    x = dt.convert_to_tensor(x)
    if axis < 0:
        axis += x.ndim + 1

    res = _OneHot(x, depth, dtype)
    res = dt.core._node_prepare(res)

    perm = list(range(res.ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    return transpose(res, perm)


def arange(
    start: int | float,
    stop: int | float | None = None,
    step: int | float = 1,
    dtype=None
):
    res = np.arange(start, stop, step)
    if res.shape[0] <= 0:
        raise ValueError(f'arange() invalid length ({res.shape[0]}) for range '
                         f'[{start}, {stop}) with step {step}')
    return dt.cast(res, dtype)


def tri(N, M=None, k=0, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype

    if M is None:
        M = N

    a = arange(N, dtype='int64')
    b = arange(-k, M-k, dtype='int64')
    m = (dt.expand_dims(a, 0) > dt.expand_dims(b, 1))

    return dt.cast(m, dtype)


def tril(m, k=0):
    m = dt.convert_to_tensor(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)

    return dt.where(mask, m, zeros(1, m.dtype))


def triu(m, k=0):
    m = dt.convert_to_tensor(m)
    mask = tri(*m.shape[-2:], k=k-1, dtype=bool)

    return dt.where(mask, zeros(1, m.dtype), m)


__all__ = [
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'broadcast_to',
    'reduce_to',
    'cast',
    'reshape',
    'squeeze',
    'expand_dims',
    'transpose',
    'swapaxes',
    'moveaxis',
    'copy',
    'flip',
    'pad',
    'slice',
    'take',
    'one_hot',
    'arange',
    'tri',
    'tril',
    'triu'
]
