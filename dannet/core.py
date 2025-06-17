from typing import Callable, Sequence, SupportsIndex, TypeAlias

import numpy as np
from dannet import lib
from dannet.lib import core, dtypes
from dannet.lib.typing import Axis, ShapeLike, TensorLike, DTypeLike
from dannet.compiler import jit


def _args_to_tensor(name: str, *args: TensorLike) -> tuple[core.BaseTensor, ...]:
    result: list[core.BaseTensor] = []
    for i, arg in enumerate(args):
        try:
            result.append(array(arg))
        except TypeError:
            msg = "{} requires ndarray or scalar arguments, got {} at position {}."
            raise TypeError(msg.format(name, type(arg), i))
    return tuple(result)


def _normalize_dtype(dtype: DTypeLike | None) -> dtypes.DannetDtype | None:
    if dtype is None:
        return dtype
    return dtypes.normalize_dtype(dtype)


def _unary_args(
    name: str,
    x: TensorLike, dtype: DTypeLike | None
) -> tuple[core.BaseTensor, dtypes.DannetDtype | None]:
    x_, = _args_to_tensor(name, x)
    dtype_ = _normalize_dtype(dtype)
    return (x_, dtype_)


def _binary_args(
    name: str,
    x1: TensorLike, x2: TensorLike,
    dtype: DTypeLike | None
) -> tuple[core.BaseTensor, core.BaseTensor, dtypes.DannetDtype | None]:
    x1, x2 = _args_to_tensor(name, x1, x2)
    shape = core._broadcast_shapes_with_name(name, x1.shape, x2.shape)
    if x1.shape != shape:
        x1 = broadcast_to(x1, shape)
    if x2.shape != shape:
        x2 = broadcast_to(x2, shape)
    dtype_ = _normalize_dtype(dtype)
    return (x1, x2, dtype_)


def _ternary_args(
    name: str,
    x1: TensorLike,
    x2: TensorLike,
    x3: TensorLike,
    dtype: DTypeLike | None
) -> tuple[core.BaseTensor, core.BaseTensor, core.BaseTensor, dtypes.DannetDtype | None]:
    x1, x2, x3 = _args_to_tensor(name, x1, x2, x3)
    shape = core._broadcast_shapes_with_name(name, x1.shape, x2.shape)
    if x1.shape != shape:
        x1 = broadcast_to(x1, shape)
    if x2.shape != shape:
        x2 = broadcast_to(x2, shape)
    if x3.shape != shape:
        x3 = broadcast_to(x3, shape)
    dtype_ = _normalize_dtype(dtype)
    return (x1, x2, x3, dtype_)


def _to_inexact(
    name: str,
    x: core.BaseTensor,
    dtype: dtypes.DannetDtype | None
) -> core.BaseTensor:
    if dtype is None:
        dtype = dtypes.promote_to_inexact(x.dtype)
    elif not dtypes.is_inexact_dtype(dtype):
        raise TypeError(f"{name}: Expected an inexact dtype, but got {dtype}")
    return x.astype(dtype)


array = core.array
broadcast_shapes = core.broadcast_shapes

# strides
py_slice = slice
_slices_type: TypeAlias = Sequence[SupportsIndex |
                                   py_slice | tuple[int, int, int]]

_as_strides_func = Callable[
    [core.BaseTensor],
    core.BaseTensor
]
_as_strides_func_shape = Callable[
    [core.BaseTensor, tuple[int, ...]],
    core.BaseTensor
]
_as_strides_func_axis = Callable[
    [core.BaseTensor, Axis],
    core.BaseTensor
]

broadcast_to_jit: _as_strides_func_shape = jit(lib.as_strides.broadcast_to)
flip_jit: _as_strides_func_axis = jit(lib.as_strides.flip)
transpose_jit: _as_strides_func_axis = jit(lib.as_strides.transpose)
expand_dims_jit: _as_strides_func_shape = jit(lib.as_strides.expand_dims)
squeeze_jit: _as_strides_func_shape = jit(lib.as_strides.squeeze)
slice_jit: Callable[
    [core.BaseTensor, _slices_type],
    core.BaseTensor
] = jit(lib.as_strides.slice)
real_jit: _as_strides_func = jit(lib.as_strides.real)
imag_jit: _as_strides_func = jit(lib.as_strides.imag)
ravel_jit: _as_strides_func = jit(lib.as_strides.ravel)
reshape_jit: _as_strides_func_shape = jit(lib.as_strides.reshape)

# unary
_unary_func = Callable[[core.BaseTensor,
                        dtypes.DannetDtype | None], core.BaseTensor]
negative_jit: _unary_func = jit(lib.unary.negative)
positive_jit: _unary_func = jit(lib.unary.positive)
abs_jit: _unary_func = jit(lib.unary.abs)
square_jit: _unary_func = jit(lib.unary.square)
sqrt_jit: _unary_func = jit(lib.unary.sqrt)
sign_jit: _unary_func = jit(lib.unary.sign)
conjugate_jit: _unary_func = jit(lib.unary.conjugate)

sin_jit: _unary_func = jit(lib.unary.sin)
cos_jit: _unary_func = jit(lib.unary.cos)
tan_jit: _unary_func = jit(lib.unary.tan)
sinh_jit: _unary_func = jit(lib.unary.sinh)
cosh_jit: _unary_func = jit(lib.unary.cosh)
tanh_jit: _unary_func = jit(lib.unary.tanh)

arcsin_jit: _unary_func = jit(lib.unary.arcsin)
arccos_jit: _unary_func = jit(lib.unary.arccos)
arctan_jit: _unary_func = jit(lib.unary.arctan)
arcsinh_jit: _unary_func = jit(lib.unary.arcsinh)
arccosh_jit: _unary_func = jit(lib.unary.arccosh)
arctanh_jit: _unary_func = jit(lib.unary.arctanh)


exp_jit: _unary_func = jit(lib.unary.exp)
exp2_jit: _unary_func = jit(lib.unary.exp2)
exp10_jit: _unary_func = jit(lib.unary.exp10)
expm1_jit: _unary_func = jit(lib.unary.expm1)

log_jit: _unary_func = jit(lib.unary.log)
log2_jit: _unary_func = jit(lib.unary.log2)
log10_jit: _unary_func = jit(lib.unary.log10)
log1p_jit: _unary_func = jit(lib.unary.log1p)

# binary
_binary_func = Callable[
    [core.BaseTensor, core.BaseTensor, dtypes.DannetDtype | None],
    core.BaseTensor
]
add_jit: _binary_func = jit(lib.binary.add)
subtract_jit: _binary_func = jit(lib.binary.subtract)
multiply_jit: _binary_func = jit(lib.binary.multiply)
divide_jit: _binary_func = jit(lib.binary.divide)
arctan2_jit: _binary_func = jit(lib.binary.arctan2)

equal_jit: _binary_func = jit(lib.binary.equal)
not_equal_jit: _binary_func = jit(lib.binary.not_equal)
less_jit: _binary_func = jit(lib.binary.less)
less_equal_jit: _binary_func = jit(lib.binary.less_equal)
greater_jit: _binary_func = jit(lib.binary.greater)
greater_equal_jit: _binary_func = jit(lib.binary.greater_equal)


# ternary
_ternary_func = Callable[
    [core.BaseTensor, core.BaseTensor, core.BaseTensor, dtypes.DannetDtype | None],
    core.BaseTensor
]
where_jit: _ternary_func = jit(lib.ternary.where)

# reductions
_reduce_func_sum = Callable[
    [core.BaseTensor, Axis, bool, dtypes.DannetDtype | None, bool],
    core.BaseTensor
]
_reduce_func_dtype = Callable[
    [core.BaseTensor, Axis, bool, dtypes.DannetDtype | None],
    core.BaseTensor
]

_reduce_func = Callable[
    [core.BaseTensor, Axis, bool],
    core.BaseTensor
]

sum_jit: _reduce_func_sum = jit(lib.reductions.sum)
mean_jit: _reduce_func_dtype = jit(lib.reductions.mean)
prod_jit: _reduce_func_dtype = jit(lib.reductions.prod)
min_jit: _reduce_func = jit(lib.reductions.min)
max_jit: _reduce_func = jit(lib.reductions.max)

# at
at_set_jit: Callable[
    [core.BaseTensor, core.BaseTensor, tuple[tuple[int, int, int], ...]],
    core.BaseTensor
] = jit(lib.at.at_set)


def negative(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("negative", x, dtype)
    return negative_jit(x, dtype_)


def positive(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("positive", x, dtype)
    if dtype_ is None or x.dtype == dtype_:
        return x
    return positive_jit(x, dtype_)


def astype(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("astype", x, dtype)
    if dtype_ is None or x.dtype == dtype:
        return x
    return positive_jit(x, dtype_)


def copy(x: TensorLike) -> core.BaseTensor:
    x, = _args_to_tensor("copy", x)
    return positive_jit(x, None)


def abs(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("abs", x, dtype)
    return abs_jit(x, None).astype(dtype_)


def absolute(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("absolute", x, dtype)
    return abs_jit(x, None).astype(dtype_)


def square(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("square", x, dtype)
    return square_jit(x, dtype_)


def sqrt(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("sqrt", x, dtype)
    return sqrt_jit(x, dtype_)


def sign(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("sign", x, dtype)
    return sign_jit(x, dtype_)


def conjugate(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("conjugate", x, dtype)
    if not dtypes.is_complex_dtype(x.dtype):
        return x.astype(dtype_)
    return conjugate_jit(x, dtype_)


def conj(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("conj", x, dtype)
    if not dtypes.is_complex_dtype(x.dtype):
        return x.astype(dtype_)
    return conjugate_jit(x, dtype_)


def sin(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("sin", x, dtype)
    return sin_jit(x, dtype_)


def cos(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("cos", x, dtype)
    return cos_jit(x, dtype_)


def tan(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("tan", x, dtype)
    return tan_jit(x, dtype_)


def sinh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("sinh", x, dtype)
    return sinh_jit(x, dtype_)


def cosh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("cosh", x, dtype)
    return cosh_jit(x, dtype_)


def tanh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("tanh", x, dtype)
    return tanh_jit(x, dtype_)


def arcsin(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arcsin", x, dtype)
    return arcsin_jit(x, dtype_)


def arccos(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arccos", x, dtype)
    return arccos_jit(x, dtype_)


def arctan(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arctan", x, dtype)
    return arctan_jit(x, dtype_)


def arcsinh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arcsinh", x, dtype)
    return arcsinh_jit(x, dtype_)


def arccosh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arccosh", x, dtype)
    return arccosh_jit(x, dtype_)


def arctanh(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("arctanh", x, dtype)
    return arctanh_jit(x, dtype_)


def exp(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("exp", x, dtype)
    return exp_jit(x, dtype_)


def exp2(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("exp2", x, dtype)
    return exp2_jit(x, dtype_)


def exp10(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("exp10", x, dtype)
    return exp10_jit(x, dtype_)


def expm1(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("expm1", x, dtype)
    return expm1_jit(x, dtype_)


def log(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("log", x, dtype)
    return log_jit(x, dtype_)


def log2(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("log2", x, dtype)
    return log2_jit(x, dtype_)


def log10(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("log10", x, dtype)
    return log10_jit(x, dtype_)


def log1p(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x, dtype_ = _unary_args("log1p", x, dtype)
    return log1p_jit(x, dtype_)


def deg2rad(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x_, dtype_ = _unary_args("deg2rad", x, dtype)
    x_ = _to_inexact("deg2rad", x_, dtype_)
    return x_ * array(np.pi / 180, dtype=x_.dtype)


def rad2deg(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x_, dtype_ = _unary_args("rad2deg", x, dtype)
    x_ = _to_inexact("rad2deg", x_, dtype_)
    return x_ * array(180 / np.pi, dtype=x_.dtype)


def radians(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x_, dtype_ = _unary_args("radians", x, dtype)
    x_ = _to_inexact("radians", x_, dtype_)
    return x_ * array(np.pi / 180, dtype=x_.dtype)


def degrees(x: TensorLike, /, dtype: DTypeLike | None = None) -> core.BaseTensor:
    x_, dtype_ = _unary_args("degrees", x, dtype)
    x_ = _to_inexact("degrees", x_, dtype_)
    return x_ * array(180 / np.pi, dtype=x_.dtype)


def angle(x: TensorLike, deg: bool = False) -> core.BaseTensor:
    x, = _args_to_tensor("angle", x)
    result = arctan2(real(x), imag(x))
    return degrees(result) if deg else result


def add(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("add", x1, x2, dtype)
    return add_jit(x1, x2, dtype_)


def subtract(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("subtract", x1, x2, dtype)
    return subtract_jit(x1, x2, dtype_)


def multiply(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("multiply", x1, x2, dtype)
    return multiply_jit(x1, x2, dtype_)


def divide(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("divide", x1, x2, dtype)
    return divide_jit(x1, x2, dtype_)


def arctan2(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("arctan2", x1, x2, dtype)
    return arctan2_jit(x1, x2, dtype_)


def equal(
    x1: TensorLike, x2: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("equal", x1, x2, dtype)
    return equal_jit(x1, x2, dtype_)


def not_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("not_equal", x1, x2, dtype)
    return not_equal_jit(x1, x2, dtype_)


def less(
    x1: TensorLike,
    x2: TensorLike,
    /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("less", x1, x2, dtype)
    return less_jit(x1, x2, dtype_)


def less_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("less_equal", x1, x2, dtype)
    return less_equal_jit(x1, x2, dtype_)


def greater(
    x1: TensorLike,
    x2: TensorLike,
    /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("greater", x1, x2, dtype)
    return greater_jit(x1, x2, dtype_)


def greater_equal(
    x1: TensorLike,
    x2: TensorLike,
    /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, dtype_ = _binary_args("greater_equal", x1, x2, dtype)
    return greater_equal_jit(x1, x2, dtype_)


def where(
    x1: TensorLike,
    x2: TensorLike,
    x3: TensorLike, /,
    dtype: DTypeLike | None = None
) -> core.BaseTensor:
    x1, x2, x3, dtype_ = _ternary_args("where", x1, x2, x3, dtype)
    return where_jit(x1, x2, x3, dtype_)


def broadcast_to(x: TensorLike, shape: ShapeLike) -> core.BaseTensor:
    x, = _args_to_tensor("broadcast_to", x)
    shape_ = lib.utils.normalize_shape(shape)
    return broadcast_to_jit(x, shape_)


def flip(x: TensorLike, axes: Axis = None) -> core.BaseTensor:
    x, = _args_to_tensor("flip", x)
    return flip_jit(x, axes)


def transpose(x: TensorLike, axes: Axis = None) -> core.BaseTensor:
    x, = _args_to_tensor("transpose", x)
    return transpose_jit(x, axes)


def expand_dims(x: TensorLike, axes: ShapeLike) -> core.BaseTensor:
    x, = _args_to_tensor("expand_dims", x)
    axes_ = lib.utils.normalize_shape(axes)
    return expand_dims_jit(x, axes_)


def squeeze(x: TensorLike, axes: ShapeLike) -> core.BaseTensor:
    x, = _args_to_tensor("squeeze", x)
    axes_ = lib.utils.normalize_shape(axes)
    return squeeze_jit(x, axes_)


def slice(x: TensorLike, slices: _slices_type) -> core.BaseTensor:
    x, = _args_to_tensor("slice", x)
    return slice_jit(x, tuple(slices))


def real(x: TensorLike) -> core.BaseTensor:
    x, = _args_to_tensor("real", x)
    return real_jit(x)


def imag(x: TensorLike) -> core.BaseTensor:
    x, = _args_to_tensor("imag", x)
    return imag_jit(x)


def ravel(x: TensorLike) -> core.BaseTensor:
    x, = _args_to_tensor("ravel", x)
    if x.strides != lib.core.default_strides(x.shape):
        x = copy(x)
    return ravel_jit(x)


def reshape(x: TensorLike, shape: ShapeLike) -> core.BaseTensor:
    x, = _args_to_tensor("reshape", x)
    try:
        shape = lib.utils.normalize_shape(shape)
    except ValueError as e:
        raise ValueError(f"reshape: {e}")

    if x.strides != lib.core.default_strides(x.shape):
        x = copy(x)
    return reshape_jit(x, shape)


def sum(
    a: TensorLike, axis: Axis = None, dtype: DTypeLike | None = None,
    keepdims: bool = False, initial: TensorLike | None = None,
    where: TensorLike | None = None, promote_integers: bool = True
) -> core.BaseTensor:
    a, = _args_to_tensor("sum", a)
    if initial is not None or where is not None:
        raise NotImplementedError("initial and where not implemented")
    return sum_jit(a, axis, keepdims, _normalize_dtype(dtype), promote_integers)


def mean(a: TensorLike, axis: Axis = None, dtype: DTypeLike | None = None,
         keepdims: bool = False, initial: TensorLike | None = None,
         where: TensorLike | None = None) -> core.BaseTensor:
    a, = _args_to_tensor("mean", a)
    if initial is not None or where is not None:
        raise NotImplementedError("initial and where not implemented")

    return mean_jit(a, axis, keepdims, _normalize_dtype(dtype))


def prod(a: TensorLike, axis: Axis = None, dtype: DTypeLike | None = None,
         keepdims: bool = False, initial: TensorLike | None = None,
         where: TensorLike | None = None) -> core.BaseTensor:
    a, = _args_to_tensor("prod", a)
    if initial is not None or where is not None:
        raise NotImplementedError("initial and where not implemented")
    return prod_jit(a, axis, keepdims, _normalize_dtype(dtype))


def min(a: TensorLike, axis: Axis = None,
        keepdims: bool = False, initial: TensorLike | None = None,
        where: TensorLike | None = None) -> core.BaseTensor:
    a, = _args_to_tensor("min", a)
    if initial is not None or where is not None:
        raise NotImplementedError("initial and where not implemented")

    return min_jit(a, axis, keepdims)


def max(a: TensorLike, axis: Axis = None,
        keepdims: bool = False, initial: TensorLike | None = None,
        where: TensorLike | None = None) -> core.BaseTensor:
    a, = _args_to_tensor("max", a)
    if initial is not None or where is not None:
        raise NotImplementedError("initial and where not implemented")

    return max_jit(a, axis, keepdims)
