from typing import Optional
import numpy as np
import dannet as dt


class RandomGenerator:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(2**64, dtype='uint64')
        self.seed = np.uint64(seed)

    def get_seed(self, n: int):
        GOLDEN_RATIO = 0x9E3779B97F4A7C15
        self.seed = np.uint64(
            (int(self.seed) + n * GOLDEN_RATIO) & 0xFFFFFFFFFFFFFFFF)
        return self.seed


default_rng = RandomGenerator()


class _RandomInt(dt.core.TensorBase):
    def __init__(self,
                 shape: dt.typing.ShapeLike,
                 rng: Optional[RandomGenerator] = None
                 ):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = 'uint64'

        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()

        if rng is None:
            rng = default_rng
        self.rng = rng
        if not isinstance(self.rng, RandomGenerator):
            raise TypeError('rng must be an instance of RandomGenerator')

    def inputs(self):
        return []

    def compute_gradients(self, grad):
        return []

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def get_config(self):
        return {'rng': self.rng}


class _RandomFloat(dt.core.TensorBase):
    def __init__(
        self,
        shape: dt.typing.ShapeLike,
        dtype: dt.typing.DTypeLike = 'float64',
        rng: Optional[RandomGenerator] = None
    ):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = dt.dtype.normalize_dtype(dtype)

        self._buffer = dt.core.TensorBuffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()

        if not dt.dtype.is_float_dtype(self.dtype) or self.dtype == 'float16':
            raise TypeError(f'RandomFloat not support {self.dtype}')

        if rng is None:
            rng = default_rng
        self.rng = rng
        if not isinstance(self.rng, RandomGenerator):
            raise TypeError('rng must be an instance of RandomGenerator')

    def inputs(self):
        return []

    def compute_gradients(self, grad):
        return []

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def get_config(self):
        return {'rng': self.rng}


def randint(
    shape: dt.typing.ShapeLike,
    rng: Optional[RandomGenerator] = None
):
    t = _RandomInt(shape, rng)
    return dt.core._node_prepare(t)


def random(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike = 'float64',
    rng: Optional[RandomGenerator] = None
):
    t = _RandomFloat(shape, dtype, rng)
    return dt.core._node_prepare(t)


def uniform(
    shape: dt.typing.ShapeLike,
    low: float = 0.0,
    high: float = 1.0,
    dtype: dt.typing.DTypeLike = 'float64',
    rng: Optional[RandomGenerator] = None
):
    return random(shape, dtype=dtype, rng=rng) * (high - low) + low


def normal(
    shape: dt.typing.ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: dt.typing.DTypeLike = 'float64',
    rng: Optional[RandomGenerator] = None
):
    # Box-Muller transform (approximate normal distribution)
    u1 = random(shape, dtype=dtype, rng=rng)
    u2 = random(shape, dtype=dtype, rng=rng)
    z0 = dt.sqrt(-2 * dt.log(u1)) * dt.cos(2 * np.pi * u2)
    return z0 * std + mean


def binomial(
    shape: dt.typing.ShapeLike,
    p: float = 0.5,
    dtype: dt.typing.DTypeLike = 'float64',
    rng: Optional[RandomGenerator] = None
):
    u = random(shape, dtype=dtype, rng=rng)
    return u < p


def truncated_normal(
    shape: dt.typing.ShapeLike,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: dt.typing.DTypeLike = 'float64',
    rng: Optional[RandomGenerator] = None
):
    lower = mean - 2 * std
    upper = mean + 2 * std
    z = normal(shape, mean, std, dtype, rng)

    return dt.clip(z, lower, upper)
