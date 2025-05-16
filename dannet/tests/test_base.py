import pytest
import numpy as np
import dannet as dt

from .utils import (
    ensure_supported,
    random_array,
    equal_output,
    dtypes
)


shapes = [(), (1,), (2, 3), (13, 34, 66)]

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
