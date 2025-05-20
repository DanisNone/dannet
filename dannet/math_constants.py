import math
import dannet as dt


inf = dt.constant(float('inf'), dt.dtype.float_dtype)
nan = dt.constant(float('nan'), dt.dtype.float_dtype)

c_pi = pi = dt.constant(math.pi, dt.dtype.float_dtype)
c_e = e = dt.constant(math.e, dt.dtype.float_dtype)

c_log2 = dt.constant(math.log(2), dt.dtype.float_dtype)
c_log10 = dt.constant(math.log(10), dt.dtype.float_dtype)

c_inv_log2 = dt.constant(1/math.log(2), dt.dtype.float_dtype)
c_inv_log10 = dt.constant(1/math.log(10), dt.dtype.float_dtype)

__all__ = [
    'inf', 'nan',
    'pi', 'e',
    'c_pi', 'c_e',
    'c_log2', 'c_log10',
    'c_inv_log2', 'c_inv_log10',
]
