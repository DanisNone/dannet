from __future__ import annotations
import abc

import dannet as dt


class _Reduce(dt.core.TensorBase):
    def __init__(self, x, axis=None, keepdims=False, dtype=None):
        self.x = dt.convert_to_tensor(x)
        axis = dt.utils.normalize_axis_tuple(axis, self.x)

        shape = []
        keepdims_shape = []
        for i, dim in enumerate(self.x.shape):
            if i in axis:
                if keepdims:
                    shape.append(1)
                keepdims_shape.append(1)
            else:
                keepdims_shape.append(dim)
                shape.append(dim)

        self._keepdims_shape = tuple(map(int, keepdims_shape))
        self._shape = tuple(map(int, shape))

        if dtype is None:
            self._dtype = self.result_type(self.x.dtype)
        else:
            self._dtype = dt.dtype.normalize_dtype(dtype)
        
        self._axis = tuple(axis)
        self._keepdims = bool(keepdims)

        self._init_default_buffer()

    @abc.abstractmethod
    def result_type(self, dtype: str) -> str:
        pass

    def inputs(self):
        return [self.x]

    def get_config(self):
        return {'axis': self._axis, 'keepdims': self._keepdims}


class _Sum(_Reduce):
    def result_type(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            return dt.dtype.int64
        if dt.dtype.is_signed_dtype(dtype):
            return dt.dtype.int64
        if dt.dtype.is_unsigned_dtype(dtype):
            return dt.dtype.uint64
        return dtype

    def _compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        return [dt.broadcast_to(grad, self.x.shape)]


class _Mean(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.promote_to_float(dtype)

    def _compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        grad = grad * dt.cast(self.size / self.x.size, grad.dtype)
        return [dt.broadcast_to(grad, self.x.shape)]


class _Prod(_Reduce):
    def result_type(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            return dt.dtype.int64
        if dt.dtype.is_signed_dtype(dtype):
            return dt.dtype.int64
        if dt.dtype.is_unsigned_dtype(dtype):
            return dt.dtype.uint64
        return dtype

    def _compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        grad = dt.broadcast_to(grad, self.x.shape)
        return [grad * self / self.x]


class _Min(_Reduce):
    def result_type(self, dtype):
        return dtype

    def _compute_gradients(self, grad):
        self_b = dt.reshape(self, self._keepdims_shape)
        grad = dt.reshape(grad, self._keepdims_shape)
        mask = dt.equal(self.x, self_b)
        return [dt.broadcast_to(grad, self.x.shape) * mask]


class _Max(_Reduce):
    def result_type(self, dtype):
        return dtype

    def _compute_gradients(self, grad):
        self_b = dt.reshape(self, self._keepdims_shape)
        grad = dt.reshape(grad, self._keepdims_shape)
        mask = dt.equal(self.x, self_b)
        return [dt.broadcast_to(grad, self.x.shape) * mask]


class _Any(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.bool_

    def _compute_gradients(self, grad):
        return None


class _All(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.bool_

    def _compute_gradients(self, grad):
        return None


class _ArgReduce(dt.core.TensorBase):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = dt.convert_to_tensor(x)

        self._keepdims = bool(keepdims)

        if axis is None:
            self._axis = None
            self.full_reduce = True
            self._shape = (1, ) * self.x.ndim if self._keepdims else ()
        else:
            axis = int(axis)
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError(
                    f'axis {axis} is out of bounds '
                    f'for tensor of dimension {x.ndim}'
                )
            self._axis = axis
            self.full_reduce = False

            output_shape = list(x._shape)
            if self._keepdims:
                output_shape[self._axis] = 1
            else:
                del output_shape[self._axis]
            self._shape = tuple(output_shape)

        self._dtype = dt.dtype.uint32

        self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def _compute_gradients(self, grad):
        return None

    def get_config(self):
        return {
            'axis': self._axis,
            'keepdims': self._keepdims
        }


class _ArgMin(_ArgReduce):
    pass


class _ArgMax(_ArgReduce):
    pass


def _make_reduce(name: str, class_: type[_Reduce]):
    def inner(x: dt.typing.TensorLike, axis=None, keepdims=False, dtype=None):
        x = dt.convert_to_tensor(x)
        y = class_(x, axis=axis, keepdims=keepdims, dtype=dtype)

        if x.size == y.size:
            y = dt.reshape(x, y.shape)
        return dt.core._node_prepare(y)
    inner.__name__ = name
    return inner


def _make_arg_reduce(name: str, class_: type[_ArgReduce]):
    def inner(x: dt.typing.TensorLike, axis=None, keepdims=False):
        x = dt.convert_to_tensor(x)
        y = class_(x, axis=axis, keepdims=keepdims)
        return dt.core._node_prepare(y)
    inner.__name__ = name
    return inner


def var(x: dt.typing.TensorLike, axis=None, keepdims=False, dtype=None):
    x = dt.convert_to_tensor(x)

    mean = dt.mean(x, axis, keepdims=True, dtype=dtype)
    variance = dt.mean(dt.square(x - mean), axis, keepdims=keepdims, dtype=dtype)
    return variance


def std(x: dt.typing.TensorLike, axis=None, keepdims=False, dtype=None):
    variance = var(x, axis=axis, keepdims=keepdims, dtype=dtype)
    return dt.sqrt(variance)


def count_nonzero(x: dt.typing.TensorLike, axis=None, keepdims=False):
    x = dt.convert_to_tensor(x)
    mask = (x != 0)
    return dt.sum(mask, axis=axis, keepdims=keepdims)



def mean(x: dt.typing.TensorLike, axis=None, keepdims=False, dtype=None):
    if dtype is not None:
        dtype = dt.dtype.normalize_dtype(dtype)
    x = dt.convert_to_tensor(x)
    
    result_dtype = dtype or x.dtype
    compute_dtype = result_dtype

    if x.dtype == dt.dtype.float16 and dtype is None:
        compute_dtype = dt.dtype.float32
    
    y = _Mean(x, axis=axis, keepdims=keepdims, dtype=compute_dtype)

    if x.size == y.size:
        y = dt.reshape(x, y.shape)

    res = dt.core._node_prepare(y)

    return res.cast(result_dtype)


sum = _make_reduce('sum', _Sum)
prod = _make_reduce('prod', _Prod)

min = _make_reduce('min', _Min)
max = _make_reduce('max', _Max)

any = _make_reduce('any', _Any)
all = _make_reduce('all', _All)

argmin = _make_arg_reduce('argmin', _ArgMin)
argmax = _make_arg_reduce('argmax', _ArgMax)

__all__ = [
    'sum',
    'mean',
    'var',
    'std',
    'prod',
    'min',
    'max',
    'any',
    'all',

    'argmin',
    'argmax',

    'count_nonzero'
]
