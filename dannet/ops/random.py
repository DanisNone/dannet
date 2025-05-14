from typing import Optional, Sequence, Tuple
import numpy as np
import dannet as dt
from dannet.core import TensorBase


class RandomGenerator:
    def __init__(self, seed=None):
        if seed is None:
            seed = np.random.randint(2**64, dtype='uint64')
        self.seed = np.uint64(seed)

    def get_seed(self, n: int):
        GOLDEN_RATIO = 0x9E3779B97F4A7C15
        self.seed = np.uint64((int(self.seed) + n * GOLDEN_RATIO) & 0xFFFFFFFFFFFFFFFF)
        return self.seed


default_rng = RandomGenerator()


class _RandomInt(dt.core.TensorBase):
    def __init__(self, shape, rng: Optional[RandomGenerator] = None):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = 'uint64'

        self._buffer = dt.core.Buffer(self)
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
    def __init__(self, shape, dtype='float64', rng: Optional[RandomGenerator] = None):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = dt.dtype.normalize_dtype(dtype)

        self._buffer = dt.core.Buffer(self)
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
    



def randint(shape, rng=None):
    t = _RandomInt(shape, rng)
    return dt.core._node_prepare(t)

def random(shape, dtype='float64', rng=None):
    t = _RandomFloat(shape, dtype, rng)
    return dt.core._node_prepare(t)

def uniform(
    shape, low=0.0, high=1.0, dtype='float64', rng: Optional[RandomGenerator] = None
):
    return random(shape, dtype=dtype, rng=rng) * (high - low) + low


def normal(
    shape, mean=0.0, std=1.0, dtype='float64', rng: Optional[RandomGenerator] = None
):
    # Box-Muller transform (approximate normal distribution)
    u1 = random(shape, dtype=dtype, rng=rng)
    u2 = random(shape, dtype=dtype, rng=rng)
    z0 = dt.sqrt(-2 * dt.log(u1)) * dt.cos(2 * np.pi * u2)
    return z0 * std + mean


def binomial(
    shape, p=0.5, dtype='float64', rng: Optional[RandomGenerator] = None
):
    u = random(shape, dtype=dtype, rng=rng)
    return dt.less(u, p)

def truncated_normal(
    shape, mean=0.0, std=1.0, dtype='float64', rng: Optional[RandomGenerator] = None
):
    def sample():
        return normal(shape, mean, std, dtype, rng)


    lower = mean - 2 * std
    upper = mean + 2 * std
    z = sample()

    return dt.clip(z, lower, upper)

    # TODO
    if dt.is_eager():
        while True:
            mask = dt.greater_equal(z, lower) * dt.less_equal(z, upper)
            if dt.equal(dt.min(mask), 1):
                break
            new_z = sample()
            z = dt.where(mask, z, new_z)
    else:
        for _ in range(4):
            mask = dt.greater_equal(z, lower) * dt.less_equal(z, upper)
            new_z = sample()
            z = dt.where(mask, z, new_z)
    return z
