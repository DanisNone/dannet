import functools
import pytest

import numpy as np
import dannet as dt

np.seterr(all='ignore')

def ensure_supported(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except dt.compiler.NotSupportDtypeError as e:
            pytest.skip(str(e))
    return test

def random_array(shape, dtype):
    if dt.dtype.is_bool_dtype(dtype):
        return np.astype(np.random.randint(2, size=shape), dtype)
    if dt.dtype.is_unsigned_dtype(dtype):
        return np.astype(np.random.randint(0, 200, size=shape), dtype)
    if dt.dtype.is_signed_dtype(dtype):
        return np.astype(np.random.randint(-100, 100, size=shape), dtype)
    if dt.dtype.is_float_dtype(dtype):
        return np.astype(np.random.uniform(-5, 5, size=shape), dtype)
    raise ValueError(f'unknown dtype: {dtype!r}')

def equal_output(x, y):
    assert x.shape == y.shape
    
    if np.promote_types(x.dtype, y.dtype) == x.dtype:
        x = x.astype(y.dtype)
    else:
        y = y.astype(x.dtype)
    if np.issubdtype(x.dtype, np.floating):
        assert np.all(np.isnan(x) == np.isnan(y))
        x = np.nan_to_num(x)
        y = np.nan_to_num(y)

        rtol, atol = {
            'float16': (1e-3, 1e-3),
            'float32': (1e-5, 1e-6),
            'float64': (1e-6, 1e-6)
        }[x.dtype.name]
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)
    else:
        np.testing.assert_array_equal(x, y)

dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']
