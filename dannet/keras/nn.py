import numpy as np
import dannet as dt

from keras.src import backend
from dannet.keras.core import cast
from dannet.keras.core import convert_to_tensor

def relu(x):
    x = convert_to_tensor(x)
    return dt.nnet.relu(x)


def relu6(x):
    x = convert_to_tensor(x)
    return dt.nnet.relu6(x)


def sigmoid(x):
    x = convert_to_tensor(x)
    return dt.nnet.sigmoid(x)


def tanh(x):
    x = convert_to_tensor(x)
    return dt.nnet.tanh(x)


def tanh_shrink(x):
    x = convert_to_tensor(x)
    return dt.nnet.tanhshrink(x)


def softplus(x):
    x = convert_to_tensor(x)
    return dt.nnet.softplus(x)


def softsign(x):
    x = convert_to_tensor(x)
    return dt.nnet.softsign(x)


def silu(x):
    x = convert_to_tensor(x)
    return dt.nnet.silu(x)


def squareplus(x, b=4):
    x = convert_to_tensor(x)
    b = convert_to_tensor(b)
    y = x + dt.sqrt(dt.square(x) + b)
    return y / 2


def log_sigmoid(x):
    x = convert_to_tensor(x)
    return dt.nnet.logsigmoid(x)


def leaky_relu(x, negative_slope=0.2):
    x = convert_to_tensor(x)
    return dt.nnet.leaky_relu(x, negative_slope=negative_slope)


def hard_sigmoid(x):
    x = convert_to_tensor(x)
    return dt.nnet.hard_sigmoid(x)


def hard_silu(x):
    x = convert_to_tensor(x)
    return dt.nnet.hard_swish(x)


def elu(x, alpha=1.0):
    x = convert_to_tensor(x)
    return dt.nnet.elu(x, alpha)


def softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = backend.standardize_dtype(x.dtype)
    if axis is None:
        output = dt.reshape(x, [-1])
        output = dt.nnet.softmax(output, axis=-1)
        output = dt.reshape(output, x.shape)
    else:
        output = dt.nnet.softmax(x, axis=axis)
    return cast(output, dtype)


def log_softmax(x, axis=-1):
    x = convert_to_tensor(x)
    dtype = backend.standardize_dtype(x.dtype)
    if axis is None:
        output = dt.reshape(x, [-1])
        output = dt.nnet.log_softmax(output, axis=-1)
        output = dt.reshape(output, x.shape)
    else:
        output = dt.nnet.log_softmax(x, axis=axis)
    return cast(output, dtype)

def one_hot(x, num_classes, axis=-1, dtype='float32', sparse=False):
    if sparse:
        raise ValueError('Unsupported value `sparse=True` with Dannet backend')

    x = convert_to_tensor(x, dtype=dt.dtype.int_dtype)
    output = dt.one_hot(x, num_classes, axis, dtype)
    return output

def categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            'Arguments `target` and `output` must have the same shape. '
            'Received: '
            f'target.shape={target.shape}, output.shape={output.shape}'
        )
    if len(target.shape) < 1:
        raise ValueError(
            'Arguments `target` and `output` must be at least rank 1. '
            'Received: '
            f'target.shape={target.shape}, output.shape={output.shape}'
        )

    if from_logits:
        log_prob = dt.nnet.log_softmax(output, axis=axis)
    else:
        output = output / dt.sum(output, axis=axis, keepdims=True)
        output = dt.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = dt.log(output)
    return -dt.sum(target * log_prob, axis=axis)

def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    target = convert_to_tensor(target, dtype=dt.dtype.uint_dtype)
    output = convert_to_tensor(output)

    if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
        target = dt.squeeze(target, axis=-1)

    if len(output.shape) < 1:
        raise ValueError(
            'Argument `output` must be at least rank 1. '
            'Received: '
            f'output.shape={output.shape}'
        )
    if target.shape != output.shape[:-1]:
        raise ValueError(
            'Arguments `target` and `output` must have the same shape '
            'up until the last dimension: '
            f'target.shape={target.shape}, output.shape={output.shape}'
        )
    if from_logits:
        log_prob = dt.nnet.log_softmax(output, axis=axis)
    else:
        output = output / dt.sum(output, axis=axis, keepdims=True)
        output = dt.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
        log_prob = dt.log(output)
    target = one_hot(target, output.shape[axis], axis=axis)
    return -dt.sum(target * log_prob, axis=axis)

def binary_crossentropy(target, output, from_logits=False):
    target = convert_to_tensor(target)
    output = convert_to_tensor(output)

    if target.shape != output.shape:
        raise ValueError(
            'Arguments `target` and `output` must have the same shape. '
            'Received: '
            f'target.shape={target.shape}, output.shape={output.shape}'
        )

    if from_logits:
        output = sigmoid(output)
    else:
        output = dt.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
    bce = target * dt.log(output) + (1.0 - target) * dt.log(1.0 - output)
    return -bce

def conv(
    inputs,
    kernel,
    strides=1,
    padding='valid',
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)

    rank = inputs.ndim - 2

    if rank == 2:
        return conv2d(inputs, kernel, strides, padding, data_format, dilation_rate)
    raise NotImplementedError(rank)


def depthwise_conv(
    inputs,
    kernel,
    strides=1,
    padding='valid',
    data_format=None,
    dilation_rate=1,
):
    inputs = convert_to_tensor(inputs)
    kernel = convert_to_tensor(kernel)

    assert kernel.shape[-1] == 1
    kernel = dt.reshape(kernel, kernel.shape[:-1])
    rank = inputs.ndim - 2

    if rank == 2:
        return depthwise_conv2d(inputs, kernel, strides, padding, data_format, dilation_rate)
    raise NotImplementedError(rank)


def conv2d(
    inputs,
    kernel,
    strides=1,
    padding='valid',
    data_format=None,
    dilation_rate=1,
):
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate)
    
    dilation_rate = tuple(dilation_rate)
    if dilation_rate != (1, 1):
        raise NotImplementedError(dilation_rate)
    
    data_format = backend.standardize_data_format(data_format)
    if data_format != 'channels_last':
        raise NotImplementedError(data_format)
    
    return dt.nnet.conv2d(inputs, kernel, strides, padding)

def depthwise_conv2d(
    inputs,
    kernel,
    strides=1,
    padding='valid',
    data_format=None,
    dilation_rate=1,
):
    if isinstance(strides, int):
        strides = (strides, strides)
    if isinstance(dilation_rate, int):
        dilation_rate = (dilation_rate, dilation_rate)
    
    dilation_rate = tuple(dilation_rate)
    if dilation_rate != (1, 1):
        raise NotImplementedError(dilation_rate)
    
    data_format = backend.standardize_data_format(data_format)
    if data_format != 'channels_last':
        raise NotImplementedError(data_format)
    
    return dt.nnet.depthwise_conv2d(inputs, kernel, strides, padding)

def batch_normalization(
    x, mean, variance, axis, offset=None, scale=None, epsilon=1e-3
):
    x = convert_to_tensor(x)
    mean = convert_to_tensor(mean)
    variance = convert_to_tensor(variance)
    
    shape = [1] * x.ndim
    shape[axis] = mean.shape[0]
    mean = dt.reshape(mean, shape)
    variance = dt.reshape(variance, shape)

    inv = dt.rsqrt(variance + epsilon)
    if scale is not None:
        scale = convert_to_tensor(scale)
        inv *= dt.reshape(scale, shape)
    
    res = (x - mean) * inv
    if offset is not None:
        offset = convert_to_tensor(offset)
        res += dt.reshape(offset, shape)
    return res

def moments(x, axes, keepdims=False, synchronized=False):
    if synchronized:
        raise NotImplementedError(
            'Argument synchronized=True is not supported with Dannet.'
        )
    x = convert_to_tensor(x)
    # The dynamic range of float16 is too limited for statistics. As a
    # workaround, we simply perform the operations on float32 and convert back
    # to float16
    need_cast = False
    ori_dtype = backend.standardize_dtype(x.dtype)
    if ori_dtype == 'float16':
        need_cast = True
        x = cast(x, 'float32')

    mean = dt.mean(x, axes, keepdims=True)

    # The variance is computed using $Var = E[|x|^2] - |E[x]|^2$, It is faster
    # but less numerically stable.
    variance = dt.mean(
        dt.square(x), axes, keepdims=True
    ) - dt.square(mean)

    if not keepdims:
        mean = dt.squeeze(mean, axes)
        variance = dt.squeeze(variance, axes)
    if need_cast:
        # avoid overflow and underflow when casting from float16 to float32
        mean = dt.clip(
            mean,
            np.finfo(np.float16).min,
            np.finfo(np.float16).max,
        )
        variance = dt.clip(
            variance,
            np.finfo(np.float16).min,
            np.finfo(np.float16).max,
        )
        mean = cast(mean, ori_dtype)
        variance = cast(variance, ori_dtype)
    return mean, variance
