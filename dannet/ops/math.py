from __future__ import annotations

import abc
from typing import Sequence
import dannet as dt
from dannet.core import TensorBase

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

class _Reciprocal(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [-grad * dt.square(self)]

class _Square(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def compute_gradients(self, grad):
        return [dt.cast(2, grad.dtype) * grad * self.x]


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
        return [grad * dt.cast(0.5, grad.dtype) / self]

class _Rsqrt(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * dt.cast(-0.5, grad.dtype) * self / self.x]

class _Exp(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * self]


class _Log(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad / self.x]


class _Sin(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * dt.cos(self.x)]

class _Cos(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [-grad * dt.sin(self.x)]

class _Tan(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * (1 + dt.square(self))]

class _Sinh(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * dt.cosh(self.x)]

class _Cosh(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * dt.sinh(self.x)]

class _Tanh(_ElementWiseUnaryFloat):
    def compute_gradients(self, grad):
        return [grad * (1 - dt.square(self))]


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
        return [zero, dt.where(self.x, grad, zero), dt.where(self.x, zero, grad)]
    
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
        
        assert self.x.ndim > 1 and self.y.ndim > 1


        if self.x._shape[-1] != self.y._shape[-2]:
            raise ValueError(f'Incompatible shapes for matmul: {self.x._shape} and {self.y._shape}')
        batch_shape = dt.utils.broadcast_shapes(self.x._shape[:-2], self.y._shape[:-2])

        self.x = dt.broadcast_to(self.x, (*batch_shape, *self.x._shape[-2:]))
        self.y = dt.broadcast_to(self.y, (*batch_shape, *self.y._shape[-2:]))
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

        if A.ndim == 1 and B.ndim == 1:
            grad_A = grad * B
            grad_B = grad * A
        elif A.ndim == 1 and B.ndim >= 2:
            grad_A = dt.matmul(grad, B, transpose_b=True)
            grad_B = dt.matmul(
                dt.reshape(A, (*A.shape, 1)), dt.reshape(grad, (1, *grad.shape))
            )
        elif A.ndim >= 2 and B.ndim == 1:
            grad_A = dt.matmul(
                dt.reshape(grad, (*grad.shape, 1)), dt.reshape(B, (1, *B.shape))
            )
            grad_B = dt.matmul(A, grad, transpose_a=True)
        else:
            grad_A = dt.matmul(grad, B, transpose_b=True)
            grad_B = dt.matmul(A, grad, transpose_a=True)

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

    if x.ndim == 0 or y.ndim == 0:
        raise ValueError('matmul: inputs must be at least 1-dimensional, got scalars')
    
    if x.ndim >= 2 and transpose_a:
        perm = list(range(x.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        x = dt.transpose(x, perm)
    
    if y.ndim >= 2 and transpose_b:
        perm = list(range(y.ndim))
        perm[-1], perm[-2] = perm[-2], perm[-1]
        y = dt.transpose(y, perm)
    
    
    x_axis = False
    if x.ndim == 1:
        x_axis = True
        x = dt.reshape(x, (1, -1))
    
    y_axis = False
    if y.ndim == 1:
        y_axis = True
        y = dt.reshape(y, (-1, 1))
    
    if x._shape[-1] != y._shape[-2]:
        raise ValueError(
            f'matmul: shapes {x._shape} and {y._shape} are incompatible: '
            f'last dim of x ({x._shape[-1]}) must match second last dim of y ({y._shape[-2]})'
        )
    
    z = _Matmul(x, y)
    z = dt.core._node_prepare(z)

    if x_axis:
        z = dt.squeeze(z, -2)
    if y_axis:
        z = dt.squeeze(z, -1)
    return z

def tensordot(x, y, axes):
    # TODO: it's a very bad implementaition

    x = dt.convert_to_tensor(x)
    y = dt.convert_to_tensor(y)

    if isinstance(axes, int):
        axes_x = tuple(range(x.ndim - axes, x.ndim))
        axes_y = tuple(range(axes))
    else:
        axes_x, axes_y = axes
    if len(axes_x) != len(axes_y):
        raise ValueError('Number of axes for contraction must be equal')
    
    for a_axis, b_axis in zip(axes_x, axes_y):
        if x.shape[a_axis] != y.shape[b_axis]:
            raise ValueError(f'Incompatible axis dimensions: x.shape[{a_axis}]={x.shape[a_axis]} != y.shape[{b_axis}]={y.shape[b_axis]}')
    

    perm = []
    for i in range(x.ndim):
        if i not in axes_x:
            perm.append(i)
    perm.extend(axes_x)
    x = dt.transpose(x, perm)

    perm = []
    for i in range(y.ndim):
        if i not in axes_y:
            perm.append(i)
    perm.extend(axes_y)
    y = dt.transpose(y, perm)
    

    x = dt.reshape(x, x.shape[:x.ndim - len(axes_x)] + (-1, ))
    y = dt.reshape(y, y.shape[:y.ndim - len(axes_y)] + (-1, ))
    
    As = x.shape[:-1]
    Bs = y.shape[:-1]
    
    x = dt.reshape(x, As + (1, ) * len(Bs) + (-1,))
    y = dt.reshape(y, (1, ) * len(As) + Bs + (-1,))
    return dt.sum(x * y, axis=-1)

def outer(x1: dt.typing.TensorLike, x2: dt.typing.TensorLike):
    x1 = dt.reshape(x1, (-1, 1))
    x2 = dt.reshape(x2, (1, -1))

    return x1 * x2


negative = _make_unary('negative', _Negative)
reciprocal = _make_unary('reciprocal', _Reciprocal)
square = _make_unary('square', _Square)
abs = _make_unary('abs', _Abs)
sign = _make_unary('sign', _Sign)
sqrt = _make_unary('sqrt', _Sqrt)
rsqrt = _make_unary('rsqrt', _Rsqrt)
exp = _make_unary('exp', _Exp)
log = _make_unary('log', _Log)

sin = _make_unary('sin', _Sin)
cos = _make_unary('cos', _Cos)
tan = _make_unary('tan', _Tan)

sinh = _make_unary('sinh', _Sinh)
cosh = _make_unary('cosh', _Cosh)
tanh = _make_unary('tanh', _Tanh)

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
    'negative', 'reciprocal', 'square', 'abs', 'sign', 'sqrt', 'rsqrt', 'exp', 'log',
    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'add', 'subtract', 'multiply', 'divide', 'power', 'minimum', 'maximum',
    'where', 'clip',
    'matmul', 'tensordot', 'outer'
]