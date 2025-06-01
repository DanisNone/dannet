raise NotImplementedError

import dannet as dt

from keras.src import tree
from keras.src.backend import floatx
from keras.src.backend.common import dtypes
from dannet.keras import convert_to_tensor


py_min = min
py_max = max


def angle(x):
    x = convert_to_tensor(x)
    return dt.angle(x)


def rot90(array, k=1, axes=(0, 1)):
    array = convert_to_tensor(array)
    return dt.rot90(array, k, axes)


def einsum(subscripts, *operands, **kwargs):
    if kwargs:
        raise ValueError(f'einsum kwargs not implemented: {kwargs}')
    operands = tuple(convert_to_tensor(op) for op in operands)
    return dt.einsum(subscripts, *operands)


def add(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.add(x1, x2)


def subtract(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.subtract(x1, x2)


def matmul(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.matmul(x1, x2)


def multiply(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.multiply(x1, x2)


def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.mean(x, axis, keepdims)


def max(x, axis=None, keepdims=False, initial=None):
    if 0 in getattr(x, 'shape', ()):
        if initial is None:
            raise ValueError('Cannot compute the max of an empty tensor.')
        elif keepdims:
            return dt.broadcast_to(initial, (1,) * len(x.shape))
        else:
            return dt.convert_to_tensor(initial)

    x = convert_to_tensor(x)
    result = dt.max(x, axis, keepdims)

    if initial is not None:
        initial = convert_to_tensor(initial, dtype=result.dtype)
        result = dt.maximum(result, initial)
    return result


def zeros(shape, dtype=None):
    if dtype is None:
        dtype = floatx()
    return dt.zeros(shape, dtype)


def zeros_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = x.dtype
    return dt.zeros(x.shape, dtype)


def ones(shape, dtype=None):
    if dtype is None:
        dtype = floatx()
    return dt.ones(shape, dtype)


def ones_like(x, dtype=None):
    x = convert_to_tensor(x)
    if dtype is None:
        dtype = x.dtype
    return dt.ones(x.shape, dtype)


def absolute(x):
    x = convert_to_tensor(x)
    return dt.abs(x)


def abs(x):
    x = convert_to_tensor(x)
    return dt.abs(x)


def all(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.all(x, axis=axis, keepdims=keepdims)


def any(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.any(x, axis=axis, keepdims=keepdims)


def amax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.max(x, axis, keepdims)


def amin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.min(x, axis, keepdims)


def append(x1, x2, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    if axis is None:
        return dt.concatenate([x1.reshape(-1), x2.reshape(-1)], axis=0)
    else:
        return dt.concatenate([x1, x2], axis=axis)


def arange(start, stop=None, step=1, dtype=None):
    return dt.arange(start, stop, step, dtype)


def arccos(x):
    x = convert_to_tensor(x)
    return dt.arccos(x)


def arccosh(x):
    x = convert_to_tensor(x)
    return dt.arccosh(x)


def arcsin(x):
    x = convert_to_tensor(x)
    return dt.arcsin(x)


def arcsinh(x):
    x = convert_to_tensor(x)
    return dt.arcsinh(x)


def arctan(x):
    x = convert_to_tensor(x)
    return dt.arctan(x)


def arctan2(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.arctan2(x1, x2)


def arctanh(x):
    x = convert_to_tensor(x)
    return dt.arctanh(x)


def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.argmax(x, axis, keepdims)


def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.argmin(x, axis, keepdims)


def argsort(x, axis=-1):
    raise NotImplementedError('argsort')


def array(x, dtype=None):
    return convert_to_tensor(x, dtype)


def average(x, axis=None, weights=None):
    x = convert_to_tensor(x)
    if weights is not None:
        weights = convert_to_tensor(weights)
        return dt.sum(x * weights, axis=axis) / dt.sum(weights, axis=axis)
    return dt.mean(x, axis=axis)


def bincount(x, weights=None, minlength=0, sparse=False):
    raise NotImplementedError('bincount')


def bitwise_and(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.bitwise_and(x, y)


def bitwise_invert(x):
    x = convert_to_tensor(x)
    return dt.bitwise_invert(x)


def bitwise_not(x):
    x = convert_to_tensor(x)
    return dt.bitwise_not(x)


def bitwise_or(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.bitwise_or(x, y)


def bitwise_xor(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.bitwise_xor(x, y)


def bitwise_left_shift(x, y):
    return left_shift(x, y)


def left_shift(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.left_shift(x, y)


def bitwise_right_shift(x, y):
    return right_shift(x, y)


def right_shift(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.right_shift(x, y)


def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return dt.broadcast_to(x, shape)


def ceil(x):
    x = convert_to_tensor(x)
    return dt.ceil(x)


def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    x_min = convert_to_tensor(x_min)
    x_max = convert_to_tensor(x_max)

    return dt.clip(x, x_min, x_max)


def concatenate(xs, axis=0):
    xs = [convert_to_tensor(x) for x in xs]
    return dt.concatenate(xs, axis=axis)


def conjugate(x):
    x = convert_to_tensor(x)
    return dt.conjugate(x)


def conj(x):
    x = convert_to_tensor(x)
    return dt.conj(x)


def copy(x):
    x = convert_to_tensor(x)
    return dt.copy(x)


def cos(x):
    x = convert_to_tensor(x)
    return dt.cos(x)


def cosh(x):
    x = convert_to_tensor(x)
    return dt.cosh(x)


def count_nonzero(x, axis=None):
    return dt.count_nonzero(x, axis)


def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=None):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.cross(x1, x2, axisa, axisb, axisc, axis)


def cumprod(x, axis=None, dtype=None):
    raise NotImplementedError('cumprod')


def cumsum(x, axis=None, dtype=None):
    raise NotImplementedError('cumsum')


def diag(x, k=0):
    x = convert_to_tensor(x)
    return dt.diag(x, k)


def diagflat(x, k=0):
    x = convert_to_tensor(x)
    return dt.diagflat(x, k)


def diagonal(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return dt.diagonal(x, offset, axis1, axis2)


def diff(a, n=1, axis=-1):
    raise NotImplementedError('diff')


def digitize(x, bins):
    raise NotImplementedError('digitize')


def dot(x, y):
    x = convert_to_tensor(x)
    y = convert_to_tensor(y)

    return dt.dot(x, y)


def empty(shape, dtype=None):
    return dt.zeros(shape, dtype)


def equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.equal(x1, x2)


def exp(x):
    x = convert_to_tensor(x)
    return dt.exp(x)


def exp2(x):
    x = convert_to_tensor(x)
    return dt.exp2(x)


def expand_dims(x, axis):
    x = convert_to_tensor(x)
    return dt.expand_dims(x, axis)


def expm1(x):
    x = convert_to_tensor(x)
    return dt.expm1(x)


def flip(x, axis=None):
    x = convert_to_tensor(x)
    return dt.flip(x, axis)


def floor(x):
    x = convert_to_tensor(x)
    return dt.floor(x)


def full(shape, fill_value, dtype=None):
    if dtype is None:
        dtype = floatx()
    fill_value = convert_to_tensor(fill_value, dtype)
    return dt.broadcast_to(fill_value, shape)


def full_like(x, fill_value, dtype=None):
    x = convert_to_tensor(x)
    dtype = dtypes.result_type(dtype or x.dtype)
    fill_value = convert_to_tensor(fill_value, dtype)

    return dt.broadcast_to(fill_value, x.shape)


def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.greater(x1, x2)


def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.greater_equal(x1, x2)


def hstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return dt.hstack(xs)


def identity(n, dtype=None):
    return eye(n, n, dtype)


def imag(x):
    x = convert_to_tensor(x)
    return dt.imag(x)


def isclose(x1, x2, rtol=1e-5, atol=1e-8, equal_nan=False):
    raise NotImplementedError('isclose')


def isfinite(x):
    raise NotImplementedError('isfinite')


def isinf(x):
    raise NotImplementedError('isinf')


def isnan(x):
    raise NotImplementedError('isnan')


def less(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.less(x1, x2)


def less_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.less_equal(x1, x2)


def linspace(
    start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0
):
    raise NotImplementedError('linspace')


def log(x):
    x = convert_to_tensor(x)
    return dt.log(x)


def log10(x):
    x = convert_to_tensor(x)
    return dt.log10(x)


def log1p(x):
    x = convert_to_tensor(x)
    return dt.log1p(x)


def log2(x):
    x = convert_to_tensor(x)
    return dt.log2(x)


def logaddexp(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.logaddexp(x1, x2)


def logical_and(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.logical_and(x1, x2)


def logical_not(x):
    x = convert_to_tensor(x)
    return dt.logical_not(x)


def logical_or(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.logical_or(x1, x2)


def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    raise NotImplementedError('logspace')


def maximum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.maximum(x1, x2)


def median(x, axis=None, keepdims=False):
    raise NotImplementedError('median')


def meshgrid(*x, indexing='xy'):
    x = [convert_to_tensor(t) for t in x]
    return dt.meshgrid(*x, indexing=indexing)


def min(x, axis=None, keepdims=False, initial=None):
    if 0 in getattr(x, 'shape', ()):
        if initial is None:
            raise ValueError('Cannot compute the min of an empty tensor.')
        elif keepdims:
            return dt.broadcast_to(initial, (1,) * len(x.shape))
        else:
            return dt.convert_to_tensor(initial)

    x = convert_to_tensor(x)
    result = dt.min(x, axis, keepdims)

    if initial is not None:
        initial = convert_to_tensor(initial, dtype=result.dtype)
        result = dt.minimum(result, initial)
    return result


def minimum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.minimum(x1, x2)


def mod(x1, x2):
    raise NotImplementedError('mod')


def moveaxis(x, source, destination):
    x = convert_to_tensor(x)
    return dt.moveaxis(x, source, destination)


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    raise NotImplementedError('nan_to_num')


def ndim(x):
    x = convert_to_tensor(x)
    return x.ndim


def nonzero(x):
    raise NotImplementedError('nonzero')


def not_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.not_equal(x1, x2)


def outer(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.outer(x1, x2)


def pad(x, pad_width, mode='constant', constant_values=None):
    if constant_values is not None:
        if mode != 'constant':
            raise ValueError(
                'Argument `constant_values` can only be '
                'provided when `mode == \'constant\'`. '
                f'Received: mode={mode}'
            )

    if mode != 'constant':
        raise NotImplementedError(mode)
    if constant_values is None:
        constant_values = 0
    if constant_values != 0:
        raise NotImplementedError(constant_values)
    x = convert_to_tensor(x)
    return dt.pad(x, pad_width)


def prod(x, axis=None, keepdims=False, dtype=None):
    x = convert_to_tensor(x)
    y = dt.prod(x, axis, keepdims)
    if dtype is not None:
        return dt.cast(y, dtype)
    return y


def quantile(x, q, axis=None, method='linear', keepdims=False):
    raise NotImplementedError('quantile')


def ravel(x):
    x = convert_to_tensor(x)
    return dt.reshape(x, -1)


def unravel_index(x, shape):
    raise NotImplementedError('unravel_index')


def real(x):
    x = convert_to_tensor(x)
    return dt.real(x)


def reciprocal(x):
    x = convert_to_tensor(x)
    return dt.reciprocal(x)


def repeat(x, repeats, axis=None):
    raise NotImplementedError('repeat')


def reshape(x, newshape):
    x = convert_to_tensor(x)
    return dt.reshape(x, newshape)


def roll(x, shift, axis=None):
    raise NotImplementedError('roll')


def searchsorted(sorted_sequence, values, side='left'):
    raise NotImplementedError('searchsorted')


def sign(x):
    x = convert_to_tensor(x)
    return dt.sign(x)


def signbit(x):
    x = convert_to_tensor(x)
    return dt.signbit(x)


def sin(x):
    x = convert_to_tensor(x)
    return dt.sin(x)


def sinh(x):
    x = convert_to_tensor(x)
    return dt.sinh(x)


def size(x):
    x = convert_to_tensor(x)
    return x.size


def sort(x, axis=-1):
    raise NotImplementedError('sort')


def split(x, indices_or_sections, axis=0):
    x = convert_to_tensor(x)
    return dt.split(x, indices_or_sections, axis)


def stack(x, axis=0):
    dtype_set = set([getattr(a, 'dtype', type(a)) for a in x])
    if len(dtype_set) > 1:
        dtype = dtypes.result_type(*dtype_set)
        x = tree.map_structure(lambda a: convert_to_tensor(a).astype(dtype), x)
    return dt.stack(x, axis=axis)  # type: ignore


def std(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.std(x, axis, keepdims)


def swapaxes(x, axis1, axis2):
    x = convert_to_tensor(x)
    return dt.swapaxes(x, axis1, axis2)


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices)

    return dt.take(x, indices, axis)


def take_along_axis(x, indices, axis=None):
    raise NotImplementedError('take_along_axis')


def tan(x):
    x = convert_to_tensor(x)
    return dt.tan(x)


def tanh(x):
    x = convert_to_tensor(x)
    return dt.tanh(x)


def tensordot(x1, x2, axes=2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.tensordot(x1, x2, axes)


def round(x, decimals=0):
    x = convert_to_tensor(x)
    return dt.round(x, decimals)


def tile(x, repeats):
    raise NotImplementedError('tile')


def trace(x, offset=0, axis1=0, axis2=1):
    x = convert_to_tensor(x)
    return dt.trace(x, offset, axis1, axis2)


def tri(N, M=None, k=0, dtype=None):
    return dt.tri(M, N, k, dtype)


def tril(x, k=0):
    x = convert_to_tensor(x)
    return dt.tril(x, k)


def triu(x, k=0):
    x = convert_to_tensor(x)
    return dt.triu(x, k)


def trunc(x):
    x = convert_to_tensor(x)
    return dt.trunc(x)


def vdot(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.vdot(x1, x2)


def inner(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.inner(x1, x2)


def vstack(xs):
    xs = [convert_to_tensor(x) for x in xs]
    return dt.vstack(xs)


def vectorize(pyfunc, *, excluded=None, signature=None):
    raise NotImplementedError('vectorize')


def where(condition, x1, x2):
    condition = convert_to_tensor(condition)
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.where(condition, x1, x2)


def divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.divide(x1, x2)


def divide_no_nan(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    res = dt.divide(x1, x2)
    return dt.where(dt.equal(x2, 0), dt.zeros_like(res), res)


def true_divide(x1, x2):
    return divide(x1, x2)


def power(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.power(x1, x2)


def negative(x):
    x = convert_to_tensor(x)
    return dt.negative(x)


def square(x):
    x = convert_to_tensor(x)
    return dt.square(x)


def sqrt(x):
    x = convert_to_tensor(x)
    return dt.sqrt(x)


def squeeze(x, axis=None):
    x = convert_to_tensor(x)
    return dt.squeeze(x, axis)


def transpose(x, axes=None):
    x = convert_to_tensor(x)
    return dt.transpose(x, axes)


def var(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.var(x, axis, keepdims)


def sum(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.sum(x, axis, keepdims)


def eye(N, M=None, k=None, dtype=None):
    return dt.eye(N=N, M=M, k=k, dtype=dtype)


def floor_divide(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.floor_divide(x1, x2)


def logical_xor(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)

    return dt.logical_xor(x1, x2)


def correlate(x1, x2, mode='valid'):
    raise NotImplementedError('correlate')


def select(condlist, choicelist, default=0):
    res = default
    for cond, choice in zip(condlist[::-1], choicelist[::-1]):
        res = where(cond, choice, res)
    return res


def slogdet(x):
    raise NotImplementedError('slogdet')


def argpartition(x, kth, axis=-1):
    raise NotImplementedError('argpartition')


def histogram(x, bins, range):
    raise NotImplementedError('histogram')


bartlett = dt.bartlett
blackman = dt.blackman
hamming = dt.hamming
