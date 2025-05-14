import functools
import pytest
import numpy as np
import dannet as dt

np.seterr(all='ignore')

shapes = [(), (1,), (2, 3), (13, 34, 66)]
dtypes = ['int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']


def ensure_supported(func):
    @functools.wraps(func)
    def test(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except dt.compiler.NotSupportDtypeError as e:
            pytest.skip(str(e))
    return test

def random_array(shape, dtype):
    if dtype == "bool":
        return np.astype(np.random.randint(2, size=shape), dtype)
    if "uint" in dtype:
        return np.astype(np.random.randint(0, 200, size=shape), dtype)
    if "int" in dtype:
        return np.astype(np.random.randint(-100, 100, size=shape), dtype)
    
    if "float" in dtype:
        return np.astype(np.random.uniform(-5, 5, size=shape), dtype)
    raise ValueError(f"unknown dtype: {dtype!r}")

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
    


unary_np = {
    'abs': np.abs,
    'cos': np.cos,
    'cosh': np.cosh,
    'exp': np.exp,
    'log': np.log,
    'negative': np.negative,
    'rsqrt': lambda x: np.sqrt(1 / x),
    'sign': np.sign,
    'sin': np.sin,
    'sinh': np.sinh,
    'sqrt': np.sqrt,
    'square': np.square,
    'tan': np.tan,
    'tanh': np.tanh
}

binary_np = {
    'add': np.add,
    'subtract': np.subtract,
    'multiply': np.multiply,
    'divide': np.divide,
    'power': np.power,
    'maximum': np.maximum,
    'minimum': np.minimum,
    'equal': np.equal,
    'greater': np.greater,
    'greater_equal': np.greater_equal,
    'less': np.less,
    'less_equal': np.less_equal,
    'not_equal': np.not_equal
}


@pytest.mark.parametrize('func', list(unary_np.keys()))
@pytest.mark.parametrize('shape', shapes)
@pytest.mark.parametrize('dtype', dtypes)
@ensure_supported
def test_unary_ops(device, func, shape, dtype):
    op = getattr(dt, func)
    x_np = random_array(shape, dtype)
    with device:
        x = dt.constant(x_np)
        y = op(x)
        y_np = y.numpy()
    
    try:
        expected = unary_np[func](x_np)
    except ValueError:
        return
    
    equal_output(expected, y_np)


@pytest.mark.parametrize('func', list(binary_np.keys()))
@pytest.mark.parametrize('shape', shapes)
@pytest.mark.parametrize('dtype', dtypes)
@ensure_supported
def test_binary_ops(device, func, shape, dtype):
    op = getattr(dt, func)
    
    a_np = random_array(shape, dtype)
    b_np = random_array(shape, dtype)
    with device:
        a = dt.constant(a_np)
        b = dt.constant(b_np)
        c = op(a, b)
        c_np = c.numpy()
    
    try:
        expected = binary_np[func](a_np, b_np)
    except ValueError:
        return
    equal_output(expected, c_np)

    
@pytest.mark.parametrize('shape', shapes)
@pytest.mark.parametrize('dtype', dtypes)
@ensure_supported
def test_reshape(device, shape, dtype):
    data_np = random_array(shape, dtype)
    with device:
        a = dt.constant(data_np)
        b = dt.reshape(a, shape)
        b_np = b.numpy()
    
    expected = np.reshape(data_np, shape)
    equal_output(expected, b_np)


@pytest.mark.parametrize('dtype', dtypes)
@ensure_supported
def test_arange(device, dtype):
    with device:
        r = dt.arange(5, dtype=dtype)
        r_np = r.numpy()
    expected = np.arange(5, dtype=dtype)

    equal_output(expected, r_np)

@pytest.mark.parametrize('shape', shapes)
@pytest.mark.parametrize('dtype', dtypes)
@ensure_supported
def test_constant_and_variable(device, shape, dtype):
    data_np = random_array(shape, dtype)
    with device:
        c = dt.constant(data_np)
        v = dt.variable(data_np)
        c_np = c.numpy()
        v_np = v.numpy()
    
    equal_output(data_np, c_np)
    equal_output(data_np, v_np)
