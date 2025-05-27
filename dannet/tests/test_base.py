from .utils import all_dtypes, get_random_array, assert_equal
import dannet as dt
import pytest

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)


skeep_zeros = [
    'divide', 'true_divide', 'floor_divide'
]

test_xfail = {
    'correctness::unary': {},
    'correctness::binary': {
        'floor_divide': ['float16'],
        'arctan2': ['complex64', 'complex128'],
        'logaddexp': ['float16'],
        'logaddexp2': ['float16']
    }
}


class TestDtype:
    @pytest.mark.parametrize('cond_dtype', all_dtypes)
    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    def test_where(self, cond_dtype, dtype1, dtype2):
        cond = jnp.array([True, False, True], dtype=cond_dtype)
        x_jax = jnp.array([1, 2, 3], dtype=dtype1)
        y_jax = jnp.array([3, 2, 1], dtype=dtype2)

        x_dt = dt.array(x_jax)
        y_dt = dt.array(y_jax)

        try:
            expected_dtype = jnp.where(cond, x_jax, y_jax).dtype
        except TypeError:
            with pytest.raises(TypeError):
                dt.function(dt.where).compute_output_spec(
                    cond, x_dt, y_dt).dtype
            return

        result_dtype = dt.function(
            dt.where).compute_output_spec(cond, x_dt, y_dt).dtype
        assert expected_dtype == result_dtype

    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    @pytest.mark.parametrize('dtype3', all_dtypes)
    def test_clip(self, dtype1, dtype2, dtype3):
        x_jax = jnp.array([1, 2, 3], dtype=dtype1)
        a_min = jnp.array([0, 1, 2], dtype=dtype2)
        a_max = jnp.array([2, 3, 4], dtype=dtype3)

        x_dt = dt.array(x_jax)
        a_min_dt = dt.array(a_min)
        a_max_dt = dt.array(a_max)

        try:
            expected_dtype = jnp.clip(x_jax, a_min, a_max).dtype
        except (TypeError, ValueError):
            with pytest.raises((TypeError, ValueError)):
                dt.function(dt.clip).compute_output_spec(
                    x_dt, a_min_dt, a_max_dt).dtype
            return

        result_dtype = dt.function(dt.clip).compute_output_spec(
            x_dt, a_min_dt, a_max_dt).dtype
        assert expected_dtype == result_dtype

    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    @pytest.mark.parametrize('op_name', [
        'add', 'subtract', 'multiply', 'divide', 'power',
        'floor_divide', 'maximum', 'minimum',
        'arctan2', 'logaddexp', 'logaddexp2',
        'equal', 'not_equal',
        'less', 'less_equal',
        'greater', 'greater_equal',

        'logical_and', 'logical_or', 'logical_xor',
        'bitwise_and', 'bitwise_or', 'bitwise_xor',

        'left_shift', 'right_shift'
    ])
    def test_binary(self, op_name, dtype1, dtype2):
        x_jax = jnp.array((1,), dtype=dtype1)
        y_jax = jnp.array((1,), dtype=dtype2)

        x_dt = dt.array((1,), dtype=dtype1)
        y_dt = dt.array((1,), dtype=dtype2)

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)

        try:
            expected_dtype = jax_op(x_jax, y_jax).dtype
        except TypeError:
            with pytest.raises(TypeError):
                result_dtype = dt.function(
                    dt_op).compute_output_spec(x_dt, y_dt).dtype
            return

        result_dtype = dt.function(dt_op).compute_output_spec(x_dt, y_dt).dtype
        assert expected_dtype == result_dtype

    @pytest.mark.parametrize('dtype', all_dtypes)
    @pytest.mark.parametrize('op_name', [
        'negative',
        'abs', 'sign', 'square', 'sqrt',
        'exp', 'exp2', 'expm1',
        'log', 'log1p', 'log2', 'log10',
        'sin', 'cos', 'tan',
        'sinh', 'cosh', 'tanh',
        'arcsin', 'arccos', 'arctan',
        'arcsinh', 'arccosh', 'arctanh',

        'round', 'ceil', 'floor', 'trunc',

        'logical_not',
        'bitwise_invert', 'bitwise_not'
    ])
    def test_unary(self, op_name, dtype):
        x_jax = jnp.array((1,), dtype=dtype)
        x_dt = dt.array((1,), dtype=dtype)

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)
        try:
            expected_dtype = jax_op(x_jax).dtype
        except TypeError:
            with pytest.raises(TypeError):
                result_dtype = dt.function(
                    dt_op).compute_output_spec(x_dt).dtype
            return

        result_dtype = dt.function(dt_op).compute_output_spec(x_dt).dtype
        assert expected_dtype == result_dtype


class TestCorrectness:
    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    def test_cast(self, dtype1, dtype2):
        x = get_random_array((10,), dtype1, dtype2)
        assert x.dtype == dtype1
        x_dt = dt.array(x)

        expected = x.astype(dtype2)
        actual = x_dt.astype(dtype2)
        assert_equal(actual, expected)

    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    @pytest.mark.parametrize('op_name', [
        'add', 'subtract', 'multiply', 'divide',
        'floor_divide', 'maximum', 'minimum',
        'arctan2', 'logaddexp', 'logaddexp2',
        'equal', 'not_equal',
        'less', 'less_equal',
        'greater', 'greater_equal',

        'logical_and', 'logical_or', 'logical_xor',
        'bitwise_and', 'bitwise_or', 'bitwise_xor',

        'left_shift', 'right_shift'
    ])
    def test_binary(self, op_name, dtype1, dtype2):
        x = get_random_array((10,), dtype1, dtype2)
        y = get_random_array((10,), dtype2, dtype1)

        if op_name in skeep_zeros:
            y = jnp.where(y == 0, jnp.ones_like(y), y)

        x_dt = dt.array(x)
        y_dt = dt.array(y)

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)

        try:
            expected = jax_op(x, y)
        except TypeError:
            with pytest.raises(TypeError):
                actual = dt_op(x_dt, y_dt)
            return

        actual = dt_op(x_dt, y_dt)

        xfail = False
        if dtype1 in test_xfail['correctness::binary'].get(op_name, []):
            xfail = True
        if dtype2 in test_xfail['correctness::binary'].get(op_name, []):
            xfail = True

        if not xfail:
            print(x)
            print(y)
            print(actual)
            print(expected)
            assert_equal(actual, expected)
            return

        try:
            assert_equal(actual, expected)
        except AssertionError:
            pytest.xfail()

    @pytest.mark.parametrize('dtype', all_dtypes)
    @pytest.mark.parametrize('op_name', [
        'negative',
        'abs', 'sign', 'square', 'sqrt',
        'exp', 'exp2', 'expm1',
        'log', 'log1p', 'log2', 'log10',
        'sin', 'cos', 'tan',
        'sinh', 'cosh', 'tanh',
        'arcsin', 'arccos', 'arctan',
        'arcsinh', 'arccosh', 'arctanh',

        'round', 'ceil', 'floor', 'trunc',

        'logical_not',
        'bitwise_invert', 'bitwise_not'
    ])
    def test_unary(self, op_name, dtype):
        x = get_random_array((10,), dtype)
        x_dt = dt.array(x)

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)

        try:
            expected = jax_op(x)
        except TypeError:
            with pytest.raises(TypeError):
                dt_op(x_dt)
            return
        actual = dt_op(x_dt)

        xfail = dtype in test_xfail['correctness::unary']

        if not xfail:
            assert_equal(actual, expected)
            return

        try:
            assert_equal(actual, expected)
        except AssertionError:
            pytest.xfail()

    @pytest.mark.parametrize('dtype1', all_dtypes)
    @pytest.mark.parametrize('dtype2', all_dtypes)
    def test_power(self, dtype1, dtype2):
        x = get_random_array((10,), dtype1, dtype2)
        y = get_random_array((10,), dtype2, dtype1)

        res_dtype = jnp.power(jnp.ones((1,), dtype1),
                              jnp.ones((1,), dtype2)).dtype
        if jnp.issubdtype(res_dtype, jnp.integer) or res_dtype == jnp.bool_:
            if x.dtype != jnp.bool_:
                x = jnp.right_shift(x, jnp.array(4, dtype1))
            assert x.dtype == dtype1

            max_power = jnp.floor(
                jnp.log(2) * (res_dtype.itemsize * 8 - 2) /
                jnp.log(jnp.maximum(x, 2))
            )
            y = jnp.where(y > max_power, jnp.ones_like(y), y)
            y = jnp.where(y < 0, jnp.zeros_like(y), y)  # type: ignore
            assert y.dtype == dtype2

        x_dt = dt.array(x)
        y_dt = dt.array(y)

        expected = jnp.power(x, y)
        actual = dt.power(x_dt, y_dt)

        assert_equal(actual, expected, tol_k=10)


reduce_ops = [
    'sum', 'prod', 'max', 'min',
    'all', 'any'
]


class TestReduce:
    @pytest.mark.parametrize('dtype', all_dtypes)
    @pytest.mark.parametrize('op_name', reduce_ops)
    @pytest.mark.parametrize('axis', [None, 0, -1, (0, 1)])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_reduce_dtype(self, op_name, dtype, axis, keepdims):
        shape = (2, 3, 4)
        x_jax = jnp.ones(shape, dtype=dtype)
        x_dt = dt.array(jnp.ones(shape, dtype=dtype))

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)

        try:
            expected = jax_op(x_jax, axis=axis, keepdims=keepdims)
            expected_dtype = expected.dtype
        except TypeError:
            with pytest.raises(TypeError):
                dt.function(dt_op).compute_output_spec(
                    x_dt, axis=axis, keepdims=keepdims)
            return

        result_spec = dt.function(dt_op).compute_output_spec(
            x_dt, axis=axis, keepdims=keepdims)
        assert result_spec.dtype == expected_dtype

    @pytest.mark.parametrize('dtype', all_dtypes)
    @pytest.mark.parametrize('op_name', reduce_ops)
    @pytest.mark.parametrize('axis', [None, 0, -1, (0, 2)])
    @pytest.mark.parametrize('keepdims', [False, True])
    def test_reduce_correctness(self, op_name, dtype, axis, keepdims):
        shape = (4, 5, 6)
        x = get_random_array(shape, dtype)
        if op_name == 'prod' and jnp.issubdtype(x.dtype, jnp.integer):
            x = jnp.clip(x, -2, 2)
            assert x.dtype == dtype
        x_dt = dt.array(x)

        jax_op = getattr(jnp, op_name)
        dt_op = getattr(dt, op_name)

        expected = jax_op(x, axis=axis, keepdims=keepdims)
        actual = dt_op(x_dt, axis=axis, keepdims=keepdims)

        try:
            assert_equal(actual, expected)
        except AssertionError as e:
            if dtype == 'float16':
                pytest.xfail()
            else:
                raise e
