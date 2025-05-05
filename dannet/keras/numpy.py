import dannet as dt
from keras.src.backend import floatx
from dannet.keras import convert_to_tensor
from keras.src.backend.common.backend_utils import canonicalize_axis
from keras.src.backend.common.backend_utils import to_tuple_or_list

py_min = min
py_max = max

def rot90(array, k=1, axes=(0, 1)):
    raise NotImplementedError('rot90')
def einsum(subscripts, *operands, **kwargs):
    raise NotImplementedError('einsum')

def add(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return dt.add(x1, x2)

def subtract(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return dt.subtract(x1, x2)

def matmul(x1, x2, transpose_a = False, transpose_b = False):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return dt.matmul(x1, x2, transpose_a, transpose_b)

def multiply(x1, x2):
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return dt.multiply(x1, x2)

def mean(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.mean(x, axis, keepdims)

def max(x, axis=None, keepdims=False, initial=None):
    if initial is not None:
        raise NotImplementedError('initial must be None')
    x = convert_to_tensor(x)
    return dt.max(x, axis, keepdims)

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
    raise NotImplementedError('all')
def any(x, axis=None, keepdims=False):
    raise NotImplementedError('any')

def amax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.max(x, axis, keepdims)

def amin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.min(x, axis, keepdims)

def append(x1, x2, axis=None):
    raise NotImplementedError('append')
def arange(start, stop=None, step=1, dtype=None):
    raise NotImplementedError('arange')
def arccos(x):
    raise NotImplementedError('arccos')
def arccosh(x):
    raise NotImplementedError('arccosh')
def arcsin(x):
    raise NotImplementedError('arcsin')
def arcsinh(x):
    raise NotImplementedError('arcsinh')
def arctan(x):
    raise NotImplementedError('arctan')
def arctan2(x1, x2):
    raise NotImplementedError('arctan2')
def arctanh(x):
    raise NotImplementedError('arctanh')

def argmax(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.argmax(x, axis, keepdims)

def argmin(x, axis=None, keepdims=False):
    x = convert_to_tensor(x)
    return dt.argmin(x, axis, keepdims)

def argsort(x, axis=-1):
    raise NotImplementedError('argsort')
def array(x, dtype=None):
    raise NotImplementedError('array')
def average(x, axis=None, weights=None):
    raise NotImplementedError('average')
def bincount(x, weights=None, minlength=0, sparse=False):
    raise NotImplementedError('bincount')
def bitwise_and(x, y):
    raise NotImplementedError('bitwise_and')
def bitwise_invert(x):
    raise NotImplementedError('bitwise_invert')
def bitwise_not(x):
    raise NotImplementedError('bitwise_not')
def bitwise_or(x, y):
    raise NotImplementedError('bitwise_or')
def bitwise_xor(x, y):
    raise NotImplementedError('bitwise_xor')
def bitwise_left_shift(x, y):
    raise NotImplementedError('bitwise_left_shift')
def left_shift(x, y):
    raise NotImplementedError('left_shift')
def bitwise_right_shift(x, y):
    raise NotImplementedError('bitwise_right_shift')
def right_shift(x, y):
    raise NotImplementedError('right_shift')

def broadcast_to(x, shape):
    x = convert_to_tensor(x)
    return dt.broadcast_to(x, shape)

def ceil(x):
    raise NotImplementedError('ceil')
def clip(x, x_min, x_max):
    x = convert_to_tensor(x)
    x_min = convert_to_tensor(x_min)
    x_max = convert_to_tensor(x_max)
    return dt.clip(x, x_min, x_max)

def concatenate(xs, axis=0):
    raise NotImplementedError('concatenate')

def conjugate(x):
    # all tensors is real only
    return convert_to_tensor(x)

def conj(x):
    # all tensors is real only
    return convert_to_tensor(x)

def copy(x):
    raise NotImplementedError('copy')

def cos(x):
    x = convert_to_tensor(x)
    return dt.cos(x)

def cosh(x):
    x = convert_to_tensor(x)
    return dt.cosh(x)

def count_nonzero(x, axis=None):
    raise NotImplementedError('count_nonzero')
def cross(x1, x2, axisa=-1, axisb=-1, axisc=-1, axis=-1):
    raise NotImplementedError('cross')
def cumprod(x, axis=None, dtype=None):
    raise NotImplementedError('cumprod')
def cumsum(x, axis=None, dtype=None):
    raise NotImplementedError('cumsum')
def diag(x, k=0):
    raise NotImplementedError('diag')
def diagflat(x, k=0):
    raise NotImplementedError('diagflat')
def diagonal(x, offset=0, axis1=0, axis2=1):
    raise NotImplementedError('diagonal')
def diff(a, n=1, axis=-1):
    raise NotImplementedError('diff')
def digitize(x, bins):
    raise NotImplementedError('digitize')

def dot(x, y):
    raise NotImplementedError('dot')

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
    raise NotImplementedError('exp2')

def expand_dims(x, axis):
    x = convert_to_tensor(x)
    return dt.expand_dims(x, axis)

def expm1(x):
    raise NotImplementedError('expm1')
def flip(x, axis=None):
    x = convert_to_tensor(x)
    return dt.flip(x, axis)

def floor(x):
    raise NotImplementedError('floor')
def full(shape, fill_value, dtype=None):
    raise NotImplementedError('full')
def full_like(x, fill_value, dtype=None):
    raise NotImplementedError('full_like')

def greater(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return dt.greater(x1, x2)

def greater_equal(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return dt.greater_equal(x1, x2)

def hstack(xs):
    raise NotImplementedError('hstack')
def identity(n, dtype=None):
    raise NotImplementedError('identity')

def imag(x):
    # all tensors is real only
    return zeros_like(x)

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
    raise NotImplementedError('log10')
def log1p(x):
    raise NotImplementedError('log1p')
def log2(x):
    raise NotImplementedError('log2')
def logaddexp(x1, x2):
    raise NotImplementedError('logaddexp')
def logical_and(x1, x2):
    raise NotImplementedError('logical_and')
def logical_not(x):
    raise NotImplementedError('logical_not')
def logical_or(x1, x2):
    raise NotImplementedError('logical_or')
def logspace(start, stop, num=50, endpoint=True, base=10, dtype=None, axis=0):
    raise NotImplementedError('logspace')

def maximum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return dt.maximum(x1, x2)

def median(x, axis=None, keepdims=False):
    raise NotImplementedError('median')
def meshgrid(*x, indexing='xy'):
    raise NotImplementedError('meshgrid')


def min(x, axis=None, keepdims=False, initial=None):
    if initial is not None:
        raise NotImplementedError('initial must be None')
    x = convert_to_tensor(x)
    return dt.min(x, axis, keepdims)

def minimum(x1, x2):
    x1 = convert_to_tensor(x1)
    x2 = convert_to_tensor(x2)
    return dt.minimum(x1, x2)

def mod(x1, x2):
    raise NotImplementedError('mod')
def moveaxis(x, source, destination):
    raise NotImplementedError('moveaxis')
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
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
    return dt.outer(x1, x2)

def pad(x, pad_width, mode='constant', constant_values=None):
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
    # all tensors is real only
    return convert_to_tensor(x)

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
    raise NotImplementedError('signbit')

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
    raise NotImplementedError('split')
def stack(x, axis=0):
    raise NotImplementedError('stack')

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
    raise NotImplementedError('tensordot')
def round(x, decimals=0):
    raise NotImplementedError('round')
def tile(x, repeats):
    raise NotImplementedError('tile')
def trace(x, offset=None, axis1=None, axis2=None):
    raise NotImplementedError('trace')
def tri(N, M=None, k=0, dtype=None):
    raise NotImplementedError('tri')
def tril(x, k=0):
    raise NotImplementedError('tril')
def triu(x, k=0):
    raise NotImplementedError('triu')
def trunc(x):
    raise NotImplementedError('trunc')
def vdot(x1, x2):
    raise NotImplementedError('vdot')
def inner(x1, x2):
    raise NotImplementedError('inner')
def vstack(xs):
    raise NotImplementedError('vstack')
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
    x1, x2 = convert_to_tensor(x1), convert_to_tensor(x2)
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
    raise NotImplementedError('eye')
def floor_divide(x1, x2):
    raise NotImplementedError('floor_divide')
def logical_xor(x1, x2):
    raise NotImplementedError('logical_xor')
def correlate(x1, x2, mode='valid'):
    raise NotImplementedError('correlate')
def select(condlist, choicelist, default=0):
    raise NotImplementedError('select')
def slogdet(x):
    raise NotImplementedError('slogdet')
def argpartition(x, kth, axis=-1):
    raise NotImplementedError('argpartition')
def histogram(x, bins, range):
    raise NotImplementedError('histogram')