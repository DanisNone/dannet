import dannet as dt


from keras.src.backend.config import floatx
from keras.src.random.seed_generator import SeedGenerator
from keras.src.random.seed_generator import draw_seed
from keras.src.random.seed_generator import make_default_seed


def dt_draw_seed(seed):
    seed = draw_seed(seed)
    if isinstance(seed, dt.core.TensorBase):
        x, y = dt.eval(seed).numpy()
        x, y = int(x), int(y)
        return (x << 32) | y
    return seed


def normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    seed = dt_draw_seed(seed)
    dtype = dtype or floatx()
    rng = dt.random.RandomGenerator(seed)
    return dt.random.normal(shape, mean, stddev, dtype, rng)


def uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    seed = dt_draw_seed(seed)
    dtype = dtype or floatx()
    rng = dt.random.RandomGenerator(seed)
    return dt.random.uniform(shape, minval, maxval, dtype, rng)

def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
    seed = dt_draw_seed(seed)
    dtype = dtype or floatx()
    rng = dt.random.RandomGenerator(seed)
    return dt.random.truncated_normal(shape, mean, stddev, dtype, rng)

def _get_concrete_noise_shape(inputs, noise_shape):
    if noise_shape is None:
        return inputs.shape

    concrete_inputs_shape = inputs.shape
    concrete_noise_shape = []
    for i, value in enumerate(noise_shape):
        concrete_noise_shape.append(
            concrete_inputs_shape[i] if value is None else value
        )
    return concrete_noise_shape


def dropout(inputs, rate, noise_shape=None, seed=None):
    seed = dt_draw_seed(seed)
    rng = dt.random.RandomGenerator(seed)

    keep_prob = 1.0 - rate

    noise_shape = _get_concrete_noise_shape(inputs, noise_shape)
    mask = dt.random.uniform(shape=noise_shape, rng=rng, dtype="float32")
    mask = dt.less(mask, keep_prob)
    mask = dt.broadcast_to(mask, inputs.shape)

    res = inputs / keep_prob
    return dt.where(mask, res, dt.zeros_like(res))