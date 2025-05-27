import jax.numpy as jnp
import jax.random as jrandom


def get_atol_rtol(dtype):
    return {
        'float16':     (1e-3, 1e-5),
        'float32':     (1e-5, 1e-8),
        'float64':     (1e-8, 1e-12),
        'complex64':   (1e-5, 1e-8),
        'complex128':  (1e-8, 1e-12),
    }[dtype.name]


def assert_equal(x, y, tol_k=1):
    x = jnp.array(x)
    y = jnp.array(y)
    assert x.shape == y.shape, (x.shape, y.shape)
    assert x.dtype == y.dtype, (x.dtype, y.dtype)

    if jnp.issubdtype(x.dtype, jnp.integer) or x.dtype == jnp.bool_:
        assert jnp.equal(x, y).all()
        return
    atol, rtol = get_atol_rtol(x.dtype)

    x = jnp.where(jnp.isinf(x), float('nan'), x)
    y = jnp.where(jnp.isinf(y), float('nan'), y)
    assert jnp.allclose(x, y, atol*tol_k, rtol*tol_k, equal_nan=True)


_random_array_num = 0


def get_random_array(shape, dtype1, dtype2=None):
    global _random_array_num
    _random_array_num += 1
    key = jrandom.PRNGKey(_random_array_num)

    dt1 = jnp.dtype(dtype1)
    dt2 = jnp.dtype(dtype2) if dtype2 is not None else None

    if dt1 == jnp.bool_:
        return jrandom.bernoulli(key, p=0.5, shape=shape)

    if jnp.issubdtype(dt1, jnp.integer):
        info1 = jnp.iinfo(dt1)
        minval, maxval = int(info1.min), int(info1.max // 2)
        if dt2 is not None and jnp.issubdtype(dt2, jnp.integer):
            info2 = jnp.iinfo(dt2)
            minval = max(int(info1.min), int(info2.min))
            maxval = min(int(info1.max // 2), int(info2.max // 2))
        return jrandom.randint(
            key, shape,
            minval=minval, maxval=maxval,
            dtype=dt1
        )

    if jnp.issubdtype(dt1, jnp.floating):
        arr = jrandom.normal(key, shape, dtype=dt1)
        if dt2 is not None and jnp.issubdtype(dt2, jnp.floating):
            max1 = jnp.finfo(dt1).max
            max2 = jnp.finfo(dt2).max
            clip_max = jnp.minimum(max1, max2)
            return jnp.clip(arr, -clip_max, clip_max).astype(dt1)

        if (
            dt2 is not None and
            jnp.issubdtype(dt2, jnp.integer) and
            jnp.iinfo(dt2).min >= 0
        ):
            info2 = jnp.iinfo(dt2)
            max_float = min(
                jnp.finfo(dt1).max,
                float(info2.max)
            )  # type: ignore
            return jnp.clip(arr, 0, max_float).astype(dt1)
        return arr.astype(dt1)

    if jnp.issubdtype(dt1, jnp.complexfloating):
        real1 = jnp.float32 if dt1 == jnp.complex64 else jnp.float64
        if dt2 is not None and jnp.issubdtype(dt2, jnp.complexfloating):
            real2 = jnp.float32 if dt2 == jnp.complex64 else jnp.float64
        elif dt2 is not None and jnp.issubdtype(dt2, jnp.floating):
            real2 = dt2
        else:
            real2 = real1

        real_part = get_random_array(shape, real1, real2)
        imag_part = get_random_array(shape, real1, real2)
        return real_part + 1j * imag_part

    raise ValueError(f'Unsupported dtype combination: {dtype1}, {dtype2}')


integers_dtypes = [
    'bool',

    'uint8',
    'uint16',
    'uint32',
    'uint64',

    'int8',
    'int16',
    'int32',
    'int64',
]

floating_dtypes = [
    'float16',
    'float32',
    'float64'
]

complex_dtypes = [
    'complex64',
    'complex128',
]


all_dtypes = integers_dtypes + floating_dtypes + complex_dtypes
dtypes_without_bool = all_dtypes.copy()
dtypes_without_bool.remove('bool')
