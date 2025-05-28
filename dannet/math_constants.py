import math
import dannet as dt


inf = dt.constant(float('inf'), dt.dtype.py_float)
nan = dt.constant(float('nan'), dt.dtype.py_float)

c_pi = pi = dt.constant(math.pi, dt.dtype.py_float)
c_e = e = dt.constant(math.e, dt.dtype.py_float)

c_log2 = dt.constant(math.log(2), dt.dtype.py_float)
c_log10 = dt.constant(math.log(10), dt.dtype.py_float)

c_inv_log2 = dt.constant(1/math.log(2), dt.dtype.py_float)
c_inv_log10 = dt.constant(1/math.log(10), dt.dtype.py_float)

__all__ = [
    'inf', 'nan',
    'pi', 'e',
    'c_pi', 'c_e',
    'c_log2', 'c_log10',
    'c_inv_log2', 'c_inv_log10',
]
