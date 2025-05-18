from __future__ import annotations
import abc

import dannet as dt


class _Reduce(dt.core.TensorBase):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = dt.convert_to_tensor(x)
        axis = dt.utils.normalize_axis_tuple(self.x, axis)

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
        self._dtype = self.result_type(self.x.dtype)

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
        return dt.dtype.max_dtype(dtype, 'uint32')

    def compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        return [dt.broadcast_to(grad, self.x.shape)]


class _Mean(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        grad = grad * dt.cast(self.size / self.x.size, grad.dtype)
        return [dt.broadcast_to(grad, self.x.shape)]


class _Prod(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.max_dtype(dtype, 'uint32')

    def compute_gradients(self, grad):
        grad = dt.reshape(grad, self._keepdims_shape)
        grad = dt.broadcast_to(grad, self.x.shape)
        return [grad * self / self.x]


class _Min(_Reduce):
    def result_type(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        self_b = dt.reshape(self, self._keepdims_shape)
        grad = dt.reshape(grad, self._keepdims_shape)
        mask = dt.equal(self.x, self_b)
        return [dt.broadcast_to(grad, self.x.shape) * mask]


class _Max(_Reduce):
    def result_type(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        self_b = dt.reshape(self, self._keepdims_shape)
        grad = dt.reshape(grad, self._keepdims_shape)
        mask = dt.equal(self.x, self_b)
        return [dt.broadcast_to(grad, self.x.shape) * mask]


class _Any(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.bool_dtype

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x)]


class _All(_Reduce):
    def result_type(self, dtype):
        return dt.dtype.bool_dtype

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x)]


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

        self._dtype = dt.dtype.uint_dtype

        self._init_default_buffer()

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x)]

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
    def inner(x: dt.typing.TensorLike, axis=None, keepdims=False):
        x = dt.convert_to_tensor(x)
        y = class_(x, axis=axis, keepdims=keepdims)

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


def var(x: dt.typing.TensorLike, axis=None, keepdims=False):
    x = dt.convert_to_tensor(x)

    mean = dt.mean(x, axis, keepdims=True)
    variance = dt.mean(dt.square(x - mean), axis, keepdims=keepdims)
    return variance


def std(x: dt.typing.TensorLike, axis=None, keepdims=False):
    variance = var(x, axis=axis, keepdims=keepdims)
    return dt.sqrt(variance)


sum = _make_reduce('sum', _Sum)
mean = _make_reduce('mean', _Mean)
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
    'argmax'
]
