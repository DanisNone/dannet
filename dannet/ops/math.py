from __future__ import annotations

import abc
import dannet as dt


class _ElementWise(dt.core.TensorBase):
    def get_config(self):
        return {}


class _ElementWiseUnary(_ElementWise):
    def __init__(self, x):
        self.x = dt.convert_to_tensor(x)
        self._shape = self.x._shape
        self._dtype = self.result_dtype(self.x._dtype)

        self._init_default_buffer()

    @abc.abstractmethod
    def result_dtype(self, dtype: str) -> str:
        pass

    def inputs(self):
        return [self.x]


class _ElementWiseUnaryFC(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dt.dtype.promote_to_float(dtype)


class _Negative(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            raise TypeError('negative does not accept dtype bool.')
        return dtype

    def _compute_gradients(self, grad):
        return [-grad]


class _Reciprocal(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def _compute_gradients(self, grad):
        return [-grad * dt.square(self)]


class _Square(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            return dt.dtype.int32
        return dtype

    def _compute_gradients(self, grad):
        return [dt.cast(2, grad.dtype) * grad * self.x]


class _Abs(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_complex_dtype(dtype):
            return dt.dtype.real_part_of_complex_dtype(dtype)
        return dtype

    def _compute_gradients(self, grad):
        return [grad * dt.sign(self.x)]


class _Sign(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            raise TypeError('sign does not accept dtype bool.')
        return dtype

    def _compute_gradients(self, grad):
        return None


class _Sqrt(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * dt.cast(0.5, grad.dtype) / self]


class _Rsqrt(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * dt.cast(-0.5, grad.dtype) * self / self.x]


class _Exp(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * self]


class _Exp2(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * self * dt.cast(dt.c_log2, grad.dtype)]


class _Exp10(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * self * dt.cast(dt.c_log10, grad.dtype)]


class _Expm1(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * (self + 1)]


class _Log(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / self.x]


class _Log2(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / self.x * dt.cast(dt.c_inv_log2, grad.dtype)]


class _Log10(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / self.x * dt.cast(dt.c_inv_log10, grad.dtype)]


class _Log1p(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / (self.x + 1)]


class _Sin(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * dt.cos(self.x)]


class _Cos(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [-grad * dt.sin(self.x)]


class _Tan(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * (1 + dt.square(self))]


class _Sinh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * dt.cosh(self.x)]


class _Cosh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * dt.sinh(self.x)]


class _Tanh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad * (1 - dt.square(self))]


class _Arcsin(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / dt.sqrt(1 - dt.square(self.x))]


class _Arccos(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [-grad / dt.sqrt(1 - dt.square(self.x))]


class _Arctan(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / (1 + dt.square(self.x))]


class _Arcsinh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / dt.sqrt(1 + dt.square(self.x))]


class _Arccosh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / dt.sqrt(-1 + dt.square(self.x))]


class _Arctanh(_ElementWiseUnaryFC):
    def _compute_gradients(self, grad):
        return [grad / (1 - dt.square(self.x))]


class _RoundBase(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_complex_dtype(dtype):
            raise TypeError(f'round operations not accept dtype {dtype}.')
        return dtype

    def _compute_gradients(self, grad):
        return None


class _Round(_RoundBase):
    def result_dtype(self, dtype):
        if dt.dtype.is_bool_dtype(dtype):
            raise TypeError('round does not accept dtype bool.')
        return dtype


class _Trunc(_RoundBase):
    pass


class _Ceil(_RoundBase):
    pass


class _Floor(_RoundBase):
    pass


class _Conjugate(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dtype

    def _compute_gradients(self, grad):
        return [dt.conjugate(grad)]


class _Real(_ElementWiseUnary):
    def result_dtype(self, dtype):
        assert dt.dtype.is_complex_dtype(dtype)
        return dt.dtype.real_part_of_complex_dtype(dtype)

    def _compute_gradients(self, grad):
        return [grad.cast(self.x.dtype)]


class _Imag(_ElementWiseUnary):
    def result_dtype(self, dtype):
        assert dt.dtype.is_complex_dtype(dtype)
        return dt.dtype.real_part_of_complex_dtype(dtype)

    def _compute_gradients(self, grad):
        return [grad.cast(self.x.dtype) * 1j]


class _ElementWiseBinary(_ElementWise):
    def __init__(self, x, y):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)

        self._shape = dt.utils.broadcast_shapes(self.x._shape, self.y._shape)
        self.x = dt.broadcast_to(self.x, self._shape)
        self.y = dt.broadcast_to(self.y, self._shape)

        self._dtype = self.result_dtype(self.x._dtype, self.y._dtype)

        self._init_default_buffer()

    @abc.abstractmethod
    def result_dtype(self, dtype1: str, dtype2: str) -> str:
        pass

    def inputs(self):
        return [self.x, self.y]


class _Add(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        return [grad, grad]


class _Subtract(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if dt.dtype.is_bool_dtype(dtype1) and dt.dtype.is_bool_dtype(dtype2):
            raise TypeError(
                'dannet boolean subtract, '
                'the `-` operator, is not supported, '
                'use the bitwise_xor, the `^` operator, '
                'or the logical_xor function instead'
            )
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        return [grad, -grad]


class _Multiply(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        return [grad * self.y, grad * self.x]


class _Divide(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        dtype = dt.dtype.promote_dtypes(dtype1, dtype2)
        return dt.dtype.promote_to_float(dtype)

    def _compute_gradients(self, grad):
        return [grad / self.y, -grad * self.x / dt.square(self.y)]


class _Power(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if dt.dtype.is_bool_dtype(dtype1) and dt.dtype.is_bool_dtype(dtype2):
            return dt.dtype.int32
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        grad_x = grad * self.y * dt.power(self.x, self.y - 1)
        grad_y = grad * dt.log(self.x) * self

        return [grad_x, grad_y]


class _Minimum(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        mask = dt.equal(self, self.x)
        grad_x = grad * mask
        grad_y = grad * dt.logical_not(mask)

        return [grad_x, grad_y]


class _Maximum(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        mask = dt.equal(self, self.x)
        grad_x = grad * mask
        grad_y = grad * dt.logical_not(mask)

        return [grad_x, grad_y]


class _FloorDivide(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if dt.dtype.is_bool_dtype(dtype1) and dt.dtype.is_bool_dtype(dtype2):
            return dt.dtype.int32
        if (
            dt.dtype.is_complex_dtype(dtype1) or
            dt.dtype.is_complex_dtype(dtype2)
        ):
            raise TypeError(
                'floor_divide does not support complex-valued inputs'
            )
        return dt.dtype.promote_dtypes(dtype1, dtype2)

    def _compute_gradients(self, grad):
        return None


class _Logaddexp(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        result = dt.dtype.promote_dtypes(dtype1, dtype2)
        return dt.dtype.promote_to_float(result)

    def _compute_gradients(self, grad):
        self_grad = grad / dt.exp(self)
        return [dt.exp(self.x) * self_grad, dt.exp(self.y) * self_grad]


class _Logaddexp2(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        result = dt.dtype.promote_dtypes(dtype1, dtype2)
        return dt.dtype.promote_to_float(result)

    def _compute_gradients(self, grad):
        self_grad = grad / dt.exp(self) * dt.c_log2
        return [dt.exp(self.x) * self_grad, dt.exp(self.y) * self_grad]


class _Arctan2(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        result = dt.dtype.promote_dtypes(dtype1, dtype2)
        return dt.dtype.promote_to_float(result)

    def _compute_gradients(self, grad):
        norm2 = dt.square(self.x) + dt.square(self.y)
        grad_norm2 = grad / norm2
        # TODO: not tested
        return [-self.y*grad_norm2, self.x*grad_norm2]


class _ElementWiseTernary(_ElementWise):
    def __init__(self, x, y, z):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)
        self.z = dt.convert_to_tensor(z)

        self._shape = dt.utils.broadcast_shapes(
            self.x._shape, self.y._shape, self.z._shape
        )

        self.x = dt.broadcast_to(self.x, self._shape)
        self.y = dt.broadcast_to(self.y, self._shape)
        self.z = dt.broadcast_to(self.z, self._shape)

        self._dtype = self.result_dtype(
            self.x._dtype, self.y._dtype, self.z._dtype
        )

        self._init_default_buffer()

    @abc.abstractmethod
    def result_dtype(self, dtype1: str, dtype2: str, dtype3: str) -> str:
        pass

    def inputs(self):
        return [self.x, self.y, self.z]


class _Where(_ElementWiseTernary):
    def result_dtype(self, dtype1, dtype2, dtype3):
        return dt.dtype.promote_dtypes(dtype2, dtype3)

    def _compute_gradients(self, grad):
        return [
            None,
            dt.where(self.x, grad, dt.zeros_like(grad)),
            dt.where(self.x, dt.zeros_like(grad), grad)
        ]


class _Clip(_ElementWiseTernary):
    def result_dtype(self, dtype1, dtype2, dtype3):
        if (
            dt.dtype.is_complex_dtype(dtype1) or
            dt.dtype.is_complex_dtype(dtype2) or
            dt.dtype.is_complex_dtype(dtype3)
        ):
            raise ValueError(
                'Clip received a complex value either through '
                'the input or the min/max keywords. '
                'Complex values have no ordering and cannot be clipped'
            )
        return dt.dtype.promote_dtypes(dtype1, dtype2, dtype3)

    def _compute_gradients(self, grad):
        condition = dt.logical_and(
            self.x >= self.y,
            self.x <= self.z
        )
        return [
            grad * condition,
            None,
            None
        ]


class _Matmul(dt.core.TensorBase):
    def __init__(self, x, y):
        self.x = dt.convert_to_tensor(x)
        self.y = dt.convert_to_tensor(y)

        assert self.x.ndim > 1 and self.y.ndim > 1

        if self.x._shape[-1] != self.y._shape[-2]:
            raise ValueError(
                f'Incompatible shapes for matmul: '
                f'{self.x._shape} and {self.y._shape}'
            )
        batch_shape = dt.utils.broadcast_shapes(
            self.x._shape[:-2], self.y._shape[:-2])

        self.x = dt.broadcast_to(self.x, (*batch_shape, *self.x._shape[-2:]))
        self.y = dt.broadcast_to(self.y, (*batch_shape, *self.y._shape[-2:]))
        self._shape = batch_shape + (self.x._shape[-2], self.y._shape[-1])

        self._dtype = dt.dtype.promote_dtypes(self.x.dtype, self.y.dtype)

        self._init_default_buffer()

    def inputs(self):
        return [self.x, self.y]

    def _compute_gradients(self, grad):
        A = self.x
        B = self.y

        if A.ndim == 1 and B.ndim == 1:
            grad_A = grad * B
            grad_B = grad * A
        elif A.ndim == 1 and B.ndim >= 2:
            grad_A = dt.matmul(grad, B, transpose_b=True)
            grad_B = dt.matmul(
                dt.reshape(A, (*A.shape, 1)), dt.reshape(grad,
                                                         (1, *grad.shape))
            )
        elif A.ndim >= 2 and B.ndim == 1:
            grad_A = dt.matmul(
                dt.reshape(grad, (*grad.shape, 1)
                           ), dt.reshape(B, (1, *B.shape))
            )
            grad_B = dt.matmul(A, grad, transpose_a=True)
        else:
            grad_A = dt.matmul(grad, B, transpose_b=True)
            grad_B = dt.matmul(A, grad, transpose_a=True)

        return [grad_A, grad_B]

    def get_config(self):
        return {}


def _make_unary(name: str, class_: type[_ElementWiseUnary]):
    def inner(x: dt.typing.TensorLike):
        y = class_(x)
        return dt.core._node_prepare(y)
    inner.__name__ = name
    return inner


def _make_round(name: str, class_: type[_RoundBase]):
    def inner(x: dt.typing.TensorLike):
        x = dt.convert_to_tensor(x)
        if dt.dtype.is_bool_dtype(x.dtype):
            return x
        if dt.dtype.is_integer_dtype(x.dtype):
            return x
        y = class_(x)
        return dt.core._node_prepare(y)
    inner.__name__ = name
    return inner


def _make_binary(
    name: str,
    class_: type[_ElementWiseBinary],
    x_neutral: bool | int | float | None = None,
    y_neutral: bool | int | float | None = None,
):
    assert (
        isinstance(x_neutral, (bool, int, float))
        or x_neutral is None
    )

    assert (
        isinstance(y_neutral, (bool, int, float))
        or y_neutral is None
    )

    def inner(x: dt.typing.TensorLike, y: dt.typing.TensorLike):
        x = dt.convert_to_tensor(x)
        y = dt.convert_to_tensor(y)
        if (
            x_neutral is not None and
            not dt.is_eager() and
            dt.core._is_constant(x)
        ):
            all_eq = dt.equal(x, x_neutral)
            all_eq = dt.all(all_eq)
            if all_eq:
                return y

        if (
            y_neutral is not None and
            not dt.is_eager() and
            dt.core._is_constant(y)
        ):
            all_eq = dt.equal(y, y_neutral)
            all_eq = dt.all(all_eq)
            all_eq = dt.eval(all_eq)
            if all_eq:
                return x

        z = class_(x, y)
        return dt.core._node_prepare(z)
    inner.__name__ = name
    return inner


def _make_ternary(name: str, class_: type[_ElementWiseTernary]):
    def inner(
            x: dt.typing.TensorLike,
            y: dt.typing.TensorLike,
            z: dt.typing.TensorLike):
        t = class_(x, y, z)
        return dt.core._node_prepare(t)
    inner.__name__ = name
    return inner


def matmul(x, y, transpose_a=False, transpose_b=False):
    x = dt.convert_to_tensor(x)
    y = dt.convert_to_tensor(y)

    if x.ndim == 0 or y.ndim == 0:
        raise ValueError(
            'matmul: inputs must be at least 1-dimensional, got scalars')

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
            f'last dim of x ({x._shape[-1]}) must match second last dim '
            f'of y ({y._shape[-2]})'
        )

    z = _Matmul(x, y)
    z = dt.core._node_prepare(z)

    if x_axis:
        z = dt.squeeze(z, -2)
    if y_axis:
        z = dt.squeeze(z, -1)
    return z


def tensordot(x, y, axes):
    x = dt.convert_to_tensor(x)
    y = dt.convert_to_tensor(y)

    if isinstance(axes, int):
        axes_x = tuple(range(x.ndim - axes, x.ndim))
        axes_y = tuple(range(axes))
    else:
        axes_x, axes_y = axes
        if isinstance(axes_x, int):
            axes_x = (axes_x, )

        if isinstance(axes_y, int):
            axes_y = (axes_y, )

    axes_x = dt.utils.normalize_axis_tuple(axes_x, x)
    axes_y = dt.utils.normalize_axis_tuple(axes_y, y)
    if len(axes_x) != len(axes_y):
        raise ValueError('Number of axes for contraction must be equal')

    for a_axis, b_axis in zip(axes_x, axes_y):
        if x.shape[a_axis] != y.shape[b_axis]:
            raise ValueError(
                f'Incompatible axis dimensions: '
                f'x.shape[{a_axis}]={x.shape[a_axis]} '
                f'!= y.shape[{b_axis}]={y.shape[b_axis]}'
            )

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

    xs = x.ndim - len(axes_x)
    ys = x.ndim - len(axes_x)
    expand_x = dt.expand_dims(x, range(xs, xs + ys))
    expand_y = dt.expand_dims(y, range(ys))
    z = expand_x * expand_y
    return dt.sum(z, axis=range(-len(axes_x), 0), dtype=z.dtype)


def dot(x, y):
    x = dt.convert_to_tensor(x)
    y = dt.convert_to_tensor(y)

    if x.ndim == 0 or y.ndim == 0:
        return x * y
    elif y.ndim == 1:
        return tensordot(x, y, axes=[[-1], [-1]])
    else:
        return tensordot(x, y, axes=[[-1], [-2]])


def vdot(x, y):
    x = dt.convert_to_tensor(x).reshape(-1)
    y = dt.convert_to_tensor(y).reshape(-1)
    return dt.sum(x * y)


def outer(x: dt.typing.TensorLike, y: dt.typing.TensorLike):
    x_arr = dt.reshape(x, (-1, 1))
    y_arr = dt.reshape(y, (1, -1))

    return x_arr * y_arr


def inner(x, y):
    return dt.tensordot(x, y, axes=(-1, -1))


def cross(
    x: dt.typing.TensorLike,
    y: dt.typing.TensorLike,
    axisa=-1, axisb=-1, axisc=-1, axis=None
):
    if axis is not None:
        axisa, axisb, axisc = (axis,) * 3

    x_arr = dt.convert_to_tensor(x)
    y_arr = dt.convert_to_tensor(y)

    if x_arr.ndim < 1 or y_arr.ndim < 1:
        raise ValueError('At least one array has zero dimension')

    axisa = dt.utils.normalize_axis_index(axisa, x_arr.ndim, 'axisa')
    axisb = dt.utils.normalize_axis_index(axisb, y_arr.ndim, 'axisb')

    x_arr = dt.moveaxis(x_arr, axisa, 0)
    y_arr = dt.moveaxis(y_arr, axisb, 0)

    x_shape = x_arr.shape
    y_shape = y_arr.shape

    if x_shape[0] not in (2, 3) or y_shape[0] not in (2, 3):
        raise ValueError(
            'incompatible dimensions for cross product (must be 2 or 3)'
        )

    out_shape = dt.utils.broadcast_shapes(x_shape[1:], y_shape[1:])

    if x_shape[0] == 3 or y_shape[0] == 3:
        out_shape = (3, ) + out_shape
        axisc = dt.utils.normalize_axis_index(axisc, len(out_shape), 'axisc')

    dtype = dt.dtype.promote_dtypes(x_arr.dtype, y_arr.dtype)
    x_arr = x_arr.astype(dtype)
    y_arr = y_arr.astype(dtype)

    x0, x1 = x_arr[0], x_arr[1]
    y0, y1 = y_arr[0], y_arr[1]

    if x_shape[0] == 2 and y_shape[0] == 2:
        return x0 * y1 - x1 * y0
    elif x_shape[0] == 2:
        assert y_shape[0] == 3
        y2 = y_arr[2]
        cp0 = x1 * y2
        cp1 = -x0 * y2
        cp2 = x0 * y1 - x1 * y0
    elif y_shape[0] == 2:
        assert x_shape[0] == 3
        x2 = x_arr[2]
        cp0 = -x2 * y1
        cp1 = x2 * y0
        cp2 = x0 * y1 - x1 * y0
    else:
        assert x_shape[0] == 3
        assert y_shape[0] == 3
        x2 = x_arr[2]
        y2 = y_arr[2]

        cp0 = x1 * y2 - x2 * y1
        cp1 = x2 * y0 - x0 * y2
        cp2 = x0 * y1 - x1 * y0

    cp = dt.stack([cp0, cp1, cp2], axis=0)
    return dt.moveaxis(cp, 0, axisc)


def round(x, decimals=0):
    if dt.dtype.is_bool_dtype(x.dtype):
        raise TypeError('round does not accept dtype bool.')

    if decimals == 0:
        return _base_round(x)
    x_dtype = x.dtype
    if dt.dtype.is_integer_dtype(x_dtype):
        if decimals > 0:
            return x
        factor = dt.cast(dt.power(10.0, decimals), dt.dtype.float32)
        x = dt.cast(x, dt.dtype.float32)
    else:
        factor = dt.cast(dt.power(10.0, decimals), x.dtype)
    x = x * factor
    x = _base_round(x)
    x = x / factor
    return dt.cast(x, x_dtype)


def conjugate(x):
    x = dt.convert_to_tensor(x)
    if not dt.dtype.is_complex_dtype(x.dtype):
        return x
    return dt.core._node_prepare(_Conjugate(x))


def real(x):
    x = dt.convert_to_tensor(x)
    if not dt.dtype.is_complex_dtype(x.dtype):
        return x
    return dt.core._node_prepare(_Real(x))


def imag(x):
    x = dt.convert_to_tensor(x)
    if not dt.dtype.is_complex_dtype(x.dtype):
        return dt.zeros_like(x)
    return dt.core._node_prepare(_Imag(x))


negative = _make_unary('negative', _Negative)
reciprocal = _make_unary('reciprocal', _Reciprocal)
square = _make_unary('square', _Square)
abs = _make_unary('abs', _Abs)
sign = _make_unary('sign', _Sign)
sqrt = _make_unary('sqrt', _Sqrt)
rsqrt = _make_unary('rsqrt', _Rsqrt)

exp = _make_unary('exp', _Exp)
exp2 = _make_unary('exp2', _Exp2)
exp10 = _make_unary('exp10', _Exp10)
expm1 = _make_unary('expm1', _Expm1)

log = _make_unary('log', _Log)
log2 = _make_unary('log2', _Log2)
log10 = _make_unary('log10', _Log10)
log1p = _make_unary('log1p', _Log1p)

sin = _make_unary('sin', _Sin)
cos = _make_unary('cos', _Cos)
tan = _make_unary('tan', _Tan)

sinh = _make_unary('sinh', _Sinh)
cosh = _make_unary('cosh', _Cosh)
tanh = _make_unary('tanh', _Tanh)

arcsin = _make_unary('arcsin', _Arcsin)
arccos = _make_unary('arccos', _Arccos)
arctan = _make_unary('arctan', _Arctan)

arcsinh = _make_unary('arcsinh', _Arcsinh)
arccosh = _make_unary('arccosh', _Arccosh)
arctanh = _make_unary('arctanh', _Arctanh)

_base_round = _make_round('base_round', _Round)
trunc = _make_round('trunc', _Trunc)
floor = _make_round('floor', _Floor)
ceil = _make_round('ceil', _Ceil)

add = _make_binary('add', _Add, 0, 0)  # x + 0 = 0 + x = 1
subtract = _make_binary('subtract', _Subtract, y_neutral=0)  # x - 0 = x
multiply = _make_binary('multiply', _Multiply, 1, 1)  # 1 * x = x * 1 = x
divide = _make_binary('divide', _Divide, y_neutral=1)  # x / 1 = x
power = _make_binary('power', _Power, y_neutral=1)  # x^1 = x
minimum = _make_binary('minimum', _Minimum)
maximum = _make_binary('maximum', _Maximum)
floor_divide = _make_binary('floor_divide', _FloorDivide)
logaddexp = _make_binary('logaddexp', _Logaddexp)
logaddexp2 = _make_binary('logaddexp2', _Logaddexp2)
arctan2 = _make_binary('arctan2', _Arctan2)

where = _make_ternary('where', _Where)
clip = _make_ternary('clip', _Clip)

conj = conjugate

__all__ = [
    'negative', 'reciprocal', 'square',
    'abs', 'sign', 'sqrt', 'rsqrt',

    'exp', 'exp2', 'exp10', 'expm1',
    'log', 'log2', 'log10', 'log1p',

    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'arcsin', 'arccos', 'arctan',
    'arcsinh', 'arccosh', 'arctanh',

    'round', 'trunc', 'floor', 'ceil',
    'add', 'subtract', 'multiply', 'divide',
    'power', 'minimum', 'maximum',
    'floor_divide',
    'logaddexp', 'logaddexp2',
    'arctan2',

    'where', 'clip',
    'matmul', 'tensordot', 'dot', 'vdot',
    'outer', 'inner',
    'cross',

    'conjugate', 'conj',
    'real', 'imag'
]
