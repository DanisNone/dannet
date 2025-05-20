import math
from typing import Sequence

import numpy as np
import dannet as dt

py_slice = slice


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
        self._is_contiguous = False

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        return [dt.reduce_to(grad, self.x._shape)]

    def get_config(self):
        return {}


class _Cast(dt.core.TensorBase):
    def __init__(self, x, new_dtype):
        self.x = dt.convert_to_tensor(x)

        self._shape = self.x._shape
        self._dtype = dt.dtype.normalize_dtype(new_dtype)

        self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        return [grad]

    def get_config(self):
        return {}


class _Bitcast(dt.core.TensorBase):
    def __init__(self, x, new_dtype):
        self.x = dt.convert_to_tensor(x)

        self._shape = self.x._shape
        self._dtype = dt.dtype.normalize_dtype(new_dtype)

        if self.x.itemsize != self.itemsize:
            raise ValueError(
                f'bitcast not possible between types '
                f'{self.x.dtype} and {self._dtype}'
            )

        self._buffer = self.x._buffer
        self._buffer_offset = self.x._buffer_offset
        self._strides = self.x._strides
        self._is_contiguous = self.x._is_contiguous

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        return None

    def get_config(self):
        return {}


class _Reshape(dt.core.TensorBase):
    def __init__(self, x, new_shape):
        self.x = dt.convert_to_tensor(x)

        if isinstance(new_shape, int):
            new_shape = [new_shape]
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
        if self.x._is_contiguous:
            self._strides = self._default_strides()
            self._buffer = self.x._buffer
            self._buffer_offset = self.x._buffer_offset
            self._is_contiguous = self.x._is_contiguous
        else:
            self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
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
        self._is_contiguous = False

        self._axes = tuple(axes)

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
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
        self._is_contiguous = False

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        return [flip(grad, self._axes)]

    def get_config(self):
        return {'axes': self._axes}


class _Copy(dt.core.TensorBase):
    def __init__(self, x):
        self.x = dt.convert_to_tensor(x)

        self._shape = self.x._shape
        self._dtype = self.x._dtype

        self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
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
        self._paddings = tuple(norm_paddings)

        if len(self._paddings) != self.x.ndim:
            raise ValueError(
                f'Paddings length {len(paddings)} '
                f'must match tensor ndim {self.x.ndim}'
            )

        self._shape = tuple(
            self.x.shape[i] + self._paddings[i][0] + self._paddings[i][1]
            for i in range(self.x.ndim)
        )

        self._dtype = self.x.dtype
        self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
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

        slices = list(slices)
        for i in range(len(slices)):
            if hasattr(slices[i], '__index__'):
                s = int(slices[i])
                slices[i] = (s, s+1, 1)
            elif isinstance(slices[i], py_slice):
                s = slices[i]
                slices[i] = (s.start, s.stop, s.step)

        self._slices = list(slices) + \
            [(None, None, None)] * (ndim - len(slices))
        self._slices = tuple(self._slices)

        orig_shape = self.x.shape
        orig_strides = self.x._strides
        new_shape = []
        new_strides = []
        offset = self.x._buffer_offset

        for i, (start, stop, step) in enumerate(self._slices):
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
                'TensorBase not support slice with zero size'
            )

        self._shape = tuple(new_shape)
        self._dtype = self.x.dtype
        self._strides = tuple(new_strides)
        self._buffer = self.x._buffer
        self._buffer_offset = offset
        self._is_contiguous = False

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
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

        self._init_default_buffer()

    def inputs(self):
        return [self.x, self.indices]

    def _compute_gradients(self, grad):
        flat_dim = math.prod(self.x.shape[1:])

        one_hot = dt.one_hot(self.indices, self.x.shape[0], dtype=grad.dtype)

        grad_2d = dt.reshape(grad, (-1, flat_dim))
        one_hot_2d = dt.reshape(one_hot, (-1, self.x.shape[0]))

        grad_x_flat = dt.matmul(one_hot_2d, grad_2d, transpose_a=True)

        grad_x = dt.reshape(grad_x_flat, self.x.shape)
        return [grad_x, None]

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

        self._init_default_buffer()

    def inputs(self):
        return [self.indices]

    def _compute_gradients(self, grad):
        return None

    def get_config(self):
        return {'depth': self._depth}


class _Concatenate(dt.core.TensorBase):
    def __init__(self, tensors):
        self._tensors = [dt.convert_to_tensor(x) for x in tensors]
        if not self._tensors:
            raise ValueError('Need at least one tensor to concatenate')

        ndim = self._tensors[0].ndim

        out_shape = list(self._tensors[0].shape)
        total = 0
        for t in self._tensors:
            if t.ndim != ndim:
                raise ValueError(
                    'All tensors must have the same number of dimensions'
                )
            for i, (d_out, d_in) in enumerate(zip(out_shape, t.shape)):
                if i == 0:
                    continue
                if d_out != d_in:
                    raise ValueError(
                        f'All dimensions except axis {0} must match: '
                        f'got {out_shape} vs {t.shape}'
                    )
            total += t.shape[0]
        out_shape[0] = total

        self._shape = tuple(out_shape)
        self._dtype = dt.dtype.max_dtype(*[inp.dtype for inp in self._tensors])

        self._init_default_buffer()

    def inputs(self):
        return self._tensors

    def _compute_gradients(self, grad):
        grads = []
        start = 0
        for t in self._tensors:
            size = t.shape[0]
            slices = []
            for i in range(grad.ndim):
                if i == 0:
                    slices.append((start, start + size, None))
                else:
                    slices.append((None, None, None))
            grads.append(slice(grad, tuple(slices)))
            start += size
        return grads

    def get_config(self):
        return {}


class _Diagonal(dt.core.TensorBase):
    def __init__(self, x, offset=0):
        self.x = dt.convert_to_tensor(x)
        if self.x.ndim < 2:
            raise ValueError(
                f'Diagonal requires input tensor with ndim>=2, '
                f'got ndim={self.x.ndim}'
            )

        dim1, dim2 = self.x.shape[-2:]

        k = offset
        if k > 0:
            diag_len = max(0, min(dim1, dim2 - k))
        else:
            diag_len = max(0, min(dim1 + k, dim2))

        new_shape = self.x.shape[:-2] + (diag_len, )

        s1, s2 = self.x._strides[-2:]
        strides = self.x._strides[:-2] + (s1 + s2, )

        buffer_offset = self.x._buffer_offset
        buffer_offset += abs(k) * (s2 if k > 0 else s1)

        self._shape = tuple(new_shape)
        self._dtype = self.x.dtype

        self._strides = tuple(strides)
        self._buffer = self.x._buffer
        self._buffer_offset = buffer_offset
        self._is_contiguous = False

        self._offset = offset

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        input_shape = self.x.shape
        mask = dt.eye(
            input_shape[-2],
            input_shape[-1],
            k=self._offset,
            dtype=dt.dtype.bool_dtype
        )

        grad_expanded = dt.expand_dims(grad, (-1, -2))
        grad_broadcasted = dt.broadcast_to(grad_expanded, input_shape)

        final_grad = dt.where(mask, grad_broadcasted, 0.0)
        return [final_grad]

    def get_config(self):
        return {
            'offset': self._offset,
        }


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


def bitcast(x: dt.typing.TensorLike, dtype: dt.typing.DTypeLike):
    x = dt.convert_to_tensor(x)
    y = _Bitcast(x, dtype)
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

    if type(axis) not in (tuple, list):
        axis = (axis,)

    out_ndim = len(axis) + x.ndim
    axis = dt.utils.normalize_axis_tuple(out_ndim, axis)

    shape_it = iter(x.shape)
    shape = [
        1 if ax in axis
        else next(shape_it)
        for ax in range(out_ndim)
    ]

    return dt.reshape(x, shape)


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


def moveaxis(x, source, destination):
    x = dt.convert_to_tensor(x)
    source = dt.utils.normalize_axis_tuple(x, source)
    destination = dt.utils.normalize_axis_tuple(x, destination)

    if len(source) != len(destination):
        raise ValueError(
            '`source` and `destination` arguments must have '
            'the same number of elements'
        )

    order = [n for n in range(x.ndim) if n not in source]

    for dest, src in sorted(zip(destination, source)):
        order.insert(dest, src)

    return transpose(x, order)


def flip(x, axis=None):
    x = dt.convert_to_tensor(x)
    y = _Flip(x, axis)
    return dt.core._node_prepare(y)


def rot90(x, k=1, axes=(0, 1)):
    x = dt.convert_to_tensor(x)

    if len(axes) != 2:
        raise ValueError('len(axes) must be 2.')

    axes = tuple(axes)
    k = int(k) % 4

    axes = tuple(ax if ax >= 0 else x.ndim + ax for ax in axes)

    if any(ax >= x.ndim for ax in axes):
        raise ValueError('Axes must be less than tensor\'s ndim.')

    if axes[0] == axes[1]:
        raise ValueError('Axes must be different.')

    perm = list(range(x.ndim))
    perm[axes[0]], perm[axes[1]] = perm[axes[1]], perm[axes[0]]

    if k == 0:
        return x
    elif k == 1:
        return flip(transpose(x, perm), axes[0])
    elif k == 2:
        return flip(x, (axes[0], axes[1]))
    else:
        return flip(transpose(x, perm), axes[1])


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


def split(x, indices_or_sections, axis=0):
    x = dt.convert_to_tensor(x)
    axis = dt.utils.normalize_axis_index(axis, x.ndim)
    L = x.shape[axis]

    if isinstance(indices_or_sections, int):
        n = indices_or_sections
        if n <= 0:
            raise ValueError('number of sections must be positive')
        if L % n != 0:
            raise ValueError(
                f'array split does not result in an equal division: '
                f'{L} elements to {n} sections'
            )
        indices = [*range(0, L, L//n), L]
    else:
        if hasattr(indices_or_sections, 'tolist'):
            indices_or_sections = indices_or_sections.tolist()
        indices = [0] + list(indices_or_sections) + [L]

    result = []
    for i in range(len(indices) - 1):
        start = indices[i]
        end = indices[i+1]

        slices = [py_slice(None)] * x.ndim
        slices[axis] = py_slice(start, end)
        print(slices, indices)
        result.append(x[tuple(slices)])

    return result


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


def concatenate(
    arrays: Sequence[dt.typing.TensorLike],
    axis=0
):
    arrays = list(arrays)
    if len(arrays) == 1:
        return dt.convert_to_tensor(arrays[0])
    for i in range(len(arrays)):
        arrays[i] = dt.moveaxis(arrays[i], axis, 0)
    res = dt.core._node_prepare(_Concatenate(arrays))
    return dt.moveaxis(res, 0, axis)


def vstack(arrays: Sequence[dt.typing.TensorLike]):
    return concatenate(arrays, axis=0)


def hstack(arrays: Sequence[dt.typing.TensorLike]):
    return concatenate(arrays, axis=1)


def stack(
    arrays: Sequence[dt.typing.TensorLike],
    axis=0
):
    arrays = [dt.convert_to_tensor(a) for a in arrays]
    expanded = [dt.expand_dims(a, axis) for a in arrays]
    return concatenate(expanded, axis=axis)


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


def eye(N: int, M: int | None = None, k: int = 0, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype
    res = np.eye(N, M, k, dtype)
    return dt.constant(res)


def tri(N, M=None, k=0, dtype=None):
    if dtype is None:
        dtype = dt.dtype.float_dtype

    if M is None:
        M = N

    a = arange(N, dtype='int64')
    b = arange(-k, M-k, dtype='int64')
    m = (dt.expand_dims(a, 1) >= dt.expand_dims(b, 0))

    return dt.cast(m, dtype)


def tril(m, k=0):
    m = dt.convert_to_tensor(m)
    mask = tri(*m.shape[-2:], k=k, dtype=bool)

    return dt.where(mask, m, zeros(1, m.dtype))


def triu(m, k=0):
    m = dt.convert_to_tensor(m)
    mask = tri(*m.shape[-2:], k=k-1, dtype=bool)

    return dt.where(mask, zeros(1, m.dtype), m)


def bartlett(M: int):
    M = int(M)
    halfM = (M - 1) / 2
    n = arange(M)
    return 1 - dt.abs(n/halfM - 1)


def blackman(M: int):
    M = int(M)
    n = arange(1-M, M, 2) * (dt.pi/(M - 1))
    return 0.42 + 0.5*dt.cos(n) + 0.08*dt.cos(2*n)


def hamming(M: int):
    M = int(M)
    n = arange(M) * (2*dt.pi/(M - 1))
    return 0.54 - 0.46*dt.cos(n)


def signbit(x):
    x = dt.convert_to_tensor(x)
    if dt.dtype.is_bool_dtype(x.dtype):
        return dt.zeros_like(x, dtype=dt.dtype.bool_dtype)
    if dt.dtype.is_integer_dtype(x.dtype):
        return x < 0

    bits = x.itemsize * 8
    uint = dt.dtype.normalize_dtype(f'uint{bits}')
    mask = 1 << (bits - 1)

    x = dt.bitwise_and(
        bitcast(x, uint),
        dt.constant(mask, uint)
    )
    return x.cast(dt.dtype.bool_dtype)


def angle(x):
    #  all tensors is real
    x = dt.convert_to_tensor(x)
    if not dt.dtype.is_float_dtype(x.dtype):
        x = dt.cast(x, dt.dtype.float_dtype)
    return dt.cast(dt.pi, x.dtype) * (x < 0)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = dt.convert_to_tensor(x)
    ndim = x.ndim
    if ndim < 2:
        raise ValueError(f'diagonal requires ndim>=2, got ndim={ndim}')

    axis1 = axis1 % ndim
    axis2 = axis2 % ndim
    if axis1 == axis2:
        raise ValueError(f'axis1 and axis2 must be different, both = {axis1}')

    perm = [i for i in range(ndim) if i not in (axis1, axis2)] + [axis1, axis2]
    x_t = x.transpose(perm)

    res = _Diagonal(x_t, offset)

    return dt.core._node_prepare(res)


def diag(x, k=0):
    x = dt.convert_to_tensor(x)
    if x.ndim == 1:
        N = x.shape[0]
        mask = dt.eye(N, dtype=dt.dtype.bool_dtype)
        zeros = dt.zeros((N, N), dtype=x.dtype)
        res = dt.where(mask, x, zeros)
        a, b = (0, k) if k > 0 else (-k, 0)
        return dt.pad(res, ((a, b), (b, a)))
    elif x.ndim == 2:
        return diagonal(x, k)
    else:
        raise ValueError('Input must be 1- or 2-d.')


def diagflat(x, k=0):
    x = dt.convert_to_tensor(x)
    flat = dt.reshape(x, (-1,))
    return diag(flat, k)


def trace(x, offset=0, axis1=0, axis2=1):
    d = diagonal(x, offset, axis1, axis2)
    return dt.sum(d, axis=-1)


def meshgrid(*xi, indexing='xy'):
    ndim = len(xi)

    if indexing not in ['xy', 'ij']:
        raise ValueError(
            'Valid values for `indexing` are \'xy\' and \'ij\'.'
        )

    s0 = (1,) * ndim
    output = [
        dt.convert_to_tensor(x).reshape(s0[:i] + (-1,) + s0[i + 1:])
        for i, x in enumerate(xi)
    ]

    if indexing == 'xy' and ndim > 1:
        output[0] = output[0].reshape((1, -1) + s0[2:])
        output[1] = output[1].reshape((-1, 1) + s0[2:])

    shape = dt.utils.broadcast_shapes(*[out.shape for out in output])
    output = [dt.broadcast_to(out, shape) for out in output]
    return output


__all__ = [
    'zeros',
    'ones',
    'zeros_like',
    'ones_like',
    'broadcast_to',
    'reduce_to',
    'cast',
    'bitcast',
    'reshape',
    'squeeze',
    'expand_dims',
    'transpose',
    'swapaxes',
    'moveaxis',
    'copy',
    'rot90',
    'flip',
    'pad',
    'slice',
    'split',

    'take',
    'one_hot',
    'arange',

    'eye',
    'tri',
    'tril',
    'triu',

    'bartlett',
    'blackman',
    'hamming',
    'angle',

    'concatenate',
    'hstack',
    'vstack',
    'stack',

    'signbit',

    'diagonal',
    'diag',
    'diagflat',
    'trace',

    'meshgrid'
]
