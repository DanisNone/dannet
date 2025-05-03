from __future__ import annotations
import abc

import dannet as dt


class _Reduce(dt.core.TensorBase):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = dt.convert_to_tensor(x)

        if axis is None:
            axis = list(range(x.ndim))
        elif isinstance(axis, int):
            axis = [axis]
        else:
            axis = list(axis)

        axis = [int(a) if a >= 0 else int(a + self.x.ndim) for a in axis]

        for a in axis:
            if not 0 <= a < self.x.ndim:
                raise ValueError(f'Invalid axis: {a}')

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

        self.axis = tuple(axis)
        self.keepdims = bool(keepdims)

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    @abc.abstractmethod
    def result_type(self, dtype: str) -> str:
        pass

    def inputs(self):
        return [self.x]
    
    def get_config(self):
        config = super(_Reduce, self).get_config()
        config['axis'] = self.axis
        config['keepdims'] = self.keepdims
        return config
    
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


class _ArgReduce(dt.core.TensorBase):
    def __init__(self, x, axis=None, keepdims=False):
        self.x = dt.convert_to_tensor(x)
        
        self.keepdims = bool(keepdims)

        if axis is None:
            self.axis = None
            self.full_reduce = True
            self._shape = (1, ) * self.x.ndim if self.keepdims else ()
        else:
            if isinstance(axis, int):
                axis = axis
            else:
                raise ValueError('axis must be an integer')
            if axis < 0:
                axis += x.ndim
            if axis < 0 or axis >= x.ndim:
                raise ValueError(f'axis {axis} is out of bounds for tensor of dimension {x.ndim}')
            self.axis = axis
            self.full_reduce = False

            output_shape = list(x._shape)
            if self.keepdims:
                output_shape[self.axis] = 1
            else:
                del output_shape[self.axis]
            self._shape = tuple(output_shape)

        self._dtype = dt.dtype.uint_dtype
        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x)]

    def get_config(self):
        config = super(_ArgReduce, self).get_config()
        config['axis'] = self.axis
        config['keepdims'] = self.keepdims

        return config
    
class _ArgMin(_ArgReduce):
    pass

class _ArgMax(_ArgReduce):
    pass


def _make_reduce(name: str, class_: type[_Reduce] | type[_ArgReduce]):
    def inner(x: dt.typing.TensorLike, axis=None, keepdims=False):
        x = dt.convert_to_tensor(x)
        y = class_(x, axis=axis, keepdims=keepdims)

        if x.size == y.size:
            y = x
        return dt.core._node_prepare(y)   
    inner.__name__ = name
    return inner

def var(x: dt.typing.TensorLike, axis=None, keepdims=False):
    x = dt.convert_to_tensor(x)

    mean = dt.mean(x, axis, keepdims=True)
    variance = dt.mean(dt.square(x - mean), axis, keepdims=keepdims)
    return variance

sum = _make_reduce('sum', _Sum)
mean = _make_reduce('mean', _Mean)
prod = _make_reduce('prod', _Prod)
min = _make_reduce('min', _Min)
max = _make_reduce('max', _Max)

argmin = _make_reduce('argmin', _ArgMin)
argmax = _make_reduce('argmax', _ArgMax)

__all__ = [
    'sum',
    'mean',
    'var',
    'prod',
    'min',
    'max',

    'argmin',
    'argmax'
]
