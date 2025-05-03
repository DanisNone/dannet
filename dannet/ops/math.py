from __future__ import annotations

import abc
from typing import Callable
import dannet as dt


class _ElementWise(dt.core.TensorBase):
    pass


class _ElementWiseUnary(_ElementWise):
    def __init__(self, x):
        self.x = dt.convert_to_tensor(x)
        self._shape = self.x._shape
        self._dtype = self.result_dtype(self.x._dtype)

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    @abc.abstractmethod
    def result_dtype(self, dtype: str) -> str:
        pass

    def inputs(self):
        return [self.x]

class _ElementWiseUnaryFloat(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)
    
class _Negative(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, 'int8')  # make signed

    def compute_gradients(self, grad):
        return [-grad]


class _Square(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        return [2 * grad * self.x]


class _Abs(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        return [grad * dt.sign(self.x)]


class _Sign(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        return [dt.zeros_like(grad)]


class _Sqrt(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * 0.5 / self]

class _Rsqrt(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * 0.5 * self / self.x]

class _Exp(_ElementWiseUnaryFloat):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        return [grad * self]


class _Log(_ElementWiseUnaryFloat):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        return [grad / self.x]


class _Sin(_ElementWiseUnaryFloat):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        return [grad * dt.cos(self.x)]


class _Cos(_ElementWiseUnaryFloat):
    def result_dtype(self, dtype):
        return dt.dtype.max_dtype(dtype, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        return [-grad * dt.sin(self.x)]


class _ElementWiseBinary(_ElementWise):
    def __init__(self, x, y):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)

        self._shape = dt.utils.broadcast_shapes(self.x._shape, self.y._shape)
        self.x = dt.broadcast_to(self.x, self._shape)
        self.y = dt.broadcast_to(self.y, self._shape)

        self._dtype = self.result_dtype(self.x._dtype, self.y._dtype)

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    @abc.abstractmethod
    def result_dtype(self, dtype1: str, dtype2: str) -> str:
        pass

    def inputs(self):
        return [self.x, self.y]

class _Add(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2, 'uint8')

    def compute_gradients(self, grad):
        return [grad, grad]


class _Subtract(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2, 'int8')  # make signed

    def compute_gradients(self, grad):
        return [grad, -grad]


class _Multiply(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2)

    def compute_gradients(self, grad):
        return [grad * self.y, grad * self.x]


class _Divide(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2, dt.dtype.float_dtype)

    def compute_gradients(self, grad):
        return [grad / self.y, -grad * self.x / dt.square(self.y)]

class _Power(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2, 'uint32')

    def compute_gradients(self, grad):
        grad_x = grad * self.y * dt.power(self.x, self.y - 1)
        grad_y = grad * dt.log(self.x) * self

        return [grad_x, grad_y]
    
class _Minimum(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2)

    def compute_gradients(self, grad):
        grad_x = grad * (dt.equal(self, self.x))
        grad_y = grad * (dt.equal(self, self.y))

        return [grad_x, grad_y]

class _Maximum(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.max_dtype(dtype1, dtype2)

    def compute_gradients(self, grad):
        grad_x = grad * (dt.equal(self, self.x))
        grad_y = grad * (dt.equal(self, self.y))

        return [grad_x, grad_y]
    
class _ElementWiseTernary(_ElementWise):
    def __init__(self, x, y, z):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)
        self.z = dt.convert_to_tensor(z)

        self._shape = dt.utils.broadcast_shapes(self.x._shape, self.y._shape, self.z._shape)
        
        self.x = dt.broadcast_to(self.x, self._shape)
        self.y = dt.broadcast_to(self.y, self._shape)
        self.z = dt.broadcast_to(self.z, self._shape)

        self._dtype = self.result_dtype(self.x._dtype, self.y._dtype, self.z._dtype)

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    @abc.abstractmethod
    def result_dtype(self, dtype1: str, dtype2: str, dtype3: str) -> str:
        pass

    def inputs(self):
        return [self.x, self.y, self.z]

class _Where(_ElementWiseTernary):
    def result_dtype(self, dtype1, dtype2, dtype3):
        return dt.dtype.max_dtype(dtype2, dtype3)

    def compute_gradients(self, grad):
        zero = dt.zeros_like(grad)
        return [dt.zeros_like(grad), dt.where(self.x, grad, zero), dt.where(self.x, zero, grad)]
    
class _Clip(_ElementWiseTernary):
    def result_dtype(self, dtype1, dtype2, dtype3):
        return dt.dtype.max_dtype(dtype1, dtype2, dtype3)

    def compute_gradients(self, grad):
        condition = dt.greater_equal(self.x, self.y) * dt.less_equal(self.x, self.z)
        return [
            grad * condition,
            dt.zeros_like(grad),
            dt.zeros_like(grad)
        ]

class _Matmul(dt.core.TensorBase):
    def __init__(self, x, y):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)
        
        if self.x.ndim == 1 and self.y.ndim == 1:
            if self.x._shape[0] != self.y._shape[0]:
                raise ValueError(f'Vector dimensions must match for dot product: {self.x._shape[0]} vs {self.y._shape[0]}')
            self._shape = ()
        elif self.x.ndim > 1 and self.y.ndim == 1:
            if self.x._shape[-1] != self.y._shape[0]:
                raise ValueError(f'Incompatible shapes for matmul: {self.x._shape} and {self.y._shape}')
            self._shape = self.x._shape[:-1]
        elif self.x.ndim == 1 and self.y.ndim > 1:
            if self.x._shape[0] != self.y._shape[-2]:
                raise ValueError(f'Incompatible shapes for matmul: {self.x._shape} and {self.y._shape}')
            self._shape = self.y._shape[:-2] + self.y._shape[-1:]
        else:
            if self.x._shape[-1] != self.y._shape[-2]:
                raise ValueError(f'Incompatible shapes for matmul: {self.x._shape} and {self.y._shape}')
            batch_shape = dt.utils.broadcast_shapes(self.x._shape[:-2], self.y._shape[:-2])

            self.x = dt.broadcast_to(x, (*batch_shape, *self.x._shape[-2:]))
            self.y = dt.broadcast_to(y, (*batch_shape, *self.y._shape[-2:]))
            self._shape = batch_shape + (self.x._shape[-2], self.y._shape[-1])
                
        self._dtype = dt.dtype.max_dtype(self.x.dtype, self.y.dtype, 'uint32')
        
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()

    def inputs(self):
        return [self.x, self.y]

    def compute_gradients(self, grad):
        A = self.x
        B = self.y

        Aperm = list(range(A.ndim))
        if A.ndim >= 2:
            Aperm[-2], Aperm[-1] = Aperm[-1], Aperm[-2]

        Bperm = list(range(B.ndim))
        if B.ndim >= 2:
            Bperm[-2], Bperm[-1] = Bperm[-1], Bperm[-2]

        if A.ndim == 1 and B.ndim == 1:
            grad_A = grad * B
            grad_B = grad * A
        elif A.ndim == 1 and B.ndim >= 2:
            grad_A = dt.matmul(grad, dt.transpose(B, perm=Bperm))
            grad_B = dt.matmul(
                dt.reshape(A, (*A.shape, 1)), dt.reshape(grad, (1, *grad.shape))
            )
        elif A.ndim >= 2 and B.ndim == 1:
            grad_A = dt.matmul(
                dt.reshape(grad, (*grad.shape, 1)), dt.reshape(B, (1, *B.shape))
            )
            grad_B = dt.matmul(dt.transpose(A, perm=Aperm), grad)
        else:
            grad_A = dt.matmul(grad, dt.transpose(B, perm=Bperm))
            grad_B = dt.matmul(dt.transpose(A, perm=Aperm), grad)

        grad_A = dt.reduce_to(grad_A, A.shape)
        grad_B = dt.reduce_to(grad_B, B.shape)
        return [grad_A, grad_B]
    
def _make_unary(name: str, class_: type[_ElementWiseUnary]):
    def inner(x: dt.typing.TensorLike):
        y = class_(x)
        return dt.core._node_prepare(y)
    inner.__name__ = name
    return inner


def _make_binary(name: str, class_: type[_ElementWiseBinary]):
    def inner(x: dt.typing.TensorLike, y: dt.typing.TensorLike):
        z = class_(x, y)
        return dt.core._node_prepare(z)
    inner.__name__ = name
    return inner

def _make_ternary(name: str, class_: type[_ElementWiseTernary]):
    def inner(x: dt.typing.TensorLike, y: dt.typing.TensorLike, z: dt.typing.TensorLike):
        t = class_(x, y, z)
        return dt.core._node_prepare(t)
    inner.__name__ = name
    return inner


def matmul(x, y, transpose_a=False, transpose_b=False):
    x = dt.convert_to_tensor(x)
    y = dt.convert_to_tensor(y)

    if x.ndim >= 2 and transpose_a:
        perm = list(range(x.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        x = dt.transpose(x, perm)
    
    if y.ndim >= 2 and transpose_b:
        perm = list(range(y.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        y = dt.transpose(y, perm)
    
    z = _Matmul(x, y)
    return dt.core._node_prepare(z)


negative = _make_unary('negative', _Negative)
square = _make_unary('square', _Square)
abs = _make_unary('abs', _Abs)
sign = _make_unary('sign', _Sign)
sqrt = _make_unary('sqrt', _Sqrt)
rsqrt = _make_unary('rsqrt', _Rsqrt)
exp = _make_unary('exp', _Exp)
log = _make_unary('log', _Log)
sin = _make_unary('sin', _Sin)
cos = _make_unary('cos', _Cos)

add = _make_binary('add', _Add)
subtract = _make_binary('subtract', _Subtract)
multiply = _make_binary('multiply', _Multiply)
divide = _make_binary('divide', _Divide)
power = _make_binary('power', _Power)
minimum = _make_binary('minimum', _Minimum)
maximum = _make_binary('maximum', _Maximum)

where = _make_ternary('where', _Where)
clip = _make_ternary('clip', _Clip)


__all__ = [
    'negative', 'square', 'abs', 'sign', 'sqrt', 'rsqrt', 'exp', 'log', 'sin', 'cos',
    'add', 'subtract', 'multiply', 'divide', 'power', 'minimum', 'maximum',
    'where', 'clip',
    'matmul'
]