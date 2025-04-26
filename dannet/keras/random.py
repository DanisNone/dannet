import dannet as dt


from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed



def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = dt.ops.random.RandomGenerator(seed)
    return dt.random.normal(shape, mean, stddev, dtype, rng)



def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = dt.ops.random.RandomGenerator(seed)
    return dt.random.uniform(shape, minval, maxval, dtype, rng)

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    dtype = dtype or floatx()
    rng = dt.ops.random.RandomGenerator(seed)
    return dt.random.truncated_normal(shape, mean, stddev, dtype, rng)
