from .utils import (
    get_random_array,
    assert_equal,
    all_dtypes
)
import dannet as dt
import random
import pytest
import jax.numpy as jnp
import jax
jax.config.update('jax_enable_x64', True)


einsum_cases = [
    'i,i->',
    'ij,j->i',
    'ij,jk->ik',
    'bij,bjk->bik',
    'qweij,qwejk->qweik',
    'ij->ji',
    'ij->',
    'ijk,ijl->kl'
]


@pytest.mark.parametrize('einsum_eq', einsum_cases)
@pytest.mark.parametrize('dtype', all_dtypes)
def test_einsum(einsum_eq, dtype):
    letters = set(einsum_eq.replace(',', '').replace('->', ''))
    dims = {letter: random.randint(1, 6) for letter in letters}

    def get_shape(term):
        return tuple(dims[ch] for ch in term)

    inputs = einsum_eq.split('->')[0].split(',')
    arrays = [get_random_array(get_shape(term), dtype) for term in inputs]

    tensors = [dt.array(arr) for arr in arrays]

    try:
        expected = jnp.einsum(einsum_eq, *arrays)
    except TypeError:
        with pytest.raises(TypeError):
            dt.einsum(einsum_eq, *tensors)
        return

    actual = dt.einsum(einsum_eq, *tensors)
    try:
        assert_equal(actual, expected)
    except AssertionError as e:
        if dtype == ['float16', 'bfloat16']:
            pytest.xfail()
        else:
            raise e


@pytest.mark.parametrize('dtype1', all_dtypes)
@pytest.mark.parametrize('dtype2', all_dtypes)
def test_matmul_dtype(dtype1, dtype2):
    x = jnp.ones((1,), dtype1)
    y = jnp.ones((1,), dtype2)

    expected_dtype = jnp.matmul(x, y).dtype
    result_dtype = dt.function(dt.matmul).compute_output_spec(x, y).dtype
    assert expected_dtype == result_dtype


@pytest.mark.parametrize('dtype', all_dtypes)
@pytest.mark.parametrize('shape_a,shape_b', [
    ((2,), (2,)),                 # dot
    ((3, 4), (4,)),               # matrix-vector
    ((3, 4), (4, 5)),             # matrix-matrix
    ((10, 3, 4), (10, 4, 5)),     # batch matmul
    ((2, 3, 3, 4), (2, 1, 4, 6)),  # broadcasted batch matmul
])
def test_matmul(dtype, shape_a, shape_b):
    x = get_random_array(shape_a, dtype)
    y = get_random_array(shape_b, dtype)

    x_dt = dt.array(x)
    y_dt = dt.array(y)

    try:
        expected = jnp.matmul(x, y)
    except TypeError:
        with pytest.raises(TypeError):
            dt.matmul(x, y)
        return

    actual = dt.matmul(x_dt, y_dt)
    try:
        assert_equal(actual, expected)
    except AssertionError as e:
        if dtype == 'float16':
            pytest.xfail()
        else:
            raise e


axes_variants = [
    None,
    1,
    ([1], [0]),
    ((0, 2), (2, 0)),
]


@pytest.mark.parametrize('dtype1', all_dtypes)
@pytest.mark.parametrize('dtype2', all_dtypes)
@pytest.mark.parametrize('axes', axes_variants)
def test_tensordot_dtype(dtype1, dtype2, axes):
    # prepare input arrays with compatible shapes
    a_shape = (2, 3, 4)
    b_shape = (4, 3, 5)
    x_jax = jnp.ones(a_shape, dtype=dtype1)
    y_jax = jnp.ones(b_shape, dtype=dtype2)
    x_dt = dt.array(x_jax)
    y_dt = dt.array(y_jax)

    def jax_op(a, b): return jnp.tensordot(a, b, axes=axes)
    def dt_op(a, b): return dt.tensordot(a, b, axes=axes)

    # expected dtype or TypeError
    try:
        expected = jax_op(x_jax, y_jax)
        expected_dtype = expected.dtype
    except (TypeError, ValueError):
        with pytest.raises((TypeError, ValueError)):
            dt.function(dt_op).compute_output_spec(x_dt, y_dt)
        return

    result_spec = dt.function(dt_op).compute_output_spec(x_dt, y_dt)
    assert result_spec.dtype == expected_dtype


@pytest.mark.parametrize('dtype1', all_dtypes)
@pytest.mark.parametrize('dtype2', all_dtypes)
@pytest.mark.parametrize('axes', axes_variants)
def test_tensordot_correctness(dtype1, dtype2, axes):
    # random input arrays
    # ensure shape compatibility based on axes
    a_shape = (2, 3, 4)
    b_shape = (4, 3, 5)
    x = get_random_array(a_shape, dtype1)
    y = get_random_array(b_shape, dtype2)

    x_dt = dt.array(x)
    y_dt = dt.array(y)

    try:
        jax_expected = jnp.tensordot(x, y, axes=axes)
    except (TypeError, ValueError):
        with pytest.raises((TypeError, ValueError)):
            dt_actual = dt.tensordot(x_dt, y_dt, axes=axes)
        return

    dt_actual = dt.tensordot(x_dt, y_dt, axes=axes)
    try:
        assert_equal(dt_actual, jax_expected, tol_k=10)
    except AssertionError as e:
        if (
            dtype1 in ['float16', 'bfloat16'] or
            dtype2 in ['float16', 'bfloat16']
        ):
            pytest.xfail()
        else:
            raise e
