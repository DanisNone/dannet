from typing import Sequence
import dannet as dt
from dannet.core import TensorBase
from dannet.ops.math import _ElementWiseUnaryFloat, _ElementWiseUnary, _make_unary



class _Relu(_ElementWiseUnary):
    # maximum(x, 0)
    def result_dtype(self, dtype):
        return dtype
    
    def compute_gradients(self, grad):
        return [grad * dt.greater(self.x, 0)]

class _Relu6(_ElementWiseUnary):
    # clip(x, 0, 6)
    def result_dtype(self, dtype):
        return dtype
    
    def compute_gradients(self, grad):
        return [grad * dt.greater(self.x, 0) * dt.less(self.x, 6)]
    
class _Sigmoid(_ElementWiseUnaryFloat):
    # 1 / (1 + exp(-x))
    def compute_gradients(self, grad):
        return [grad * self * (1 - self)]

class _Tanh(_ElementWiseUnaryFloat):
    # tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    # = 1 - 2 / (1 + exp(2 x))
    def compute_gradients(self, grad):
        return [grad * (1 - dt.square(self))]

class _Softplus(_ElementWiseUnaryFloat):
    # log(1 + exp(x))
    def compute_gradients(self, grad):
        return [grad * sigmoid(self.x)]

class _Softsign(_ElementWiseUnaryFloat):
    # x / (1 + abs(x))
    def compute_gradients(self, grad):
        return [grad / dt.square(1 + dt.abs(self.x))]
    
class _HardSigmoid(_ElementWiseUnaryFloat):
    # clip(x / 6 + 0.5, 0, 1)
    def compute_gradients(self, grad):
        return [dt.where(dt.greater_equal(self.x, -3) * dt.less_equal(self.x, 3), grad * (1 / 6), 0)]
    
relu = _make_unary('relu', _Relu)
relu6 = _make_unary('relu6', _Relu6)
sigmoid = _make_unary('sigmoid', _Sigmoid)
tanh = _make_unary('tanh', _Tanh)
softplus = _make_unary('softplus', _Softplus)
softsign = _make_unary('softsign', _Softsign)
hard_sigmoid = _make_unary('hard_sigmoid', _HardSigmoid)

def tanhshrink(x):
    x = dt.convert_to_tensor(x)
    return x - tanh(x)

def silu(x):
    x = dt.convert_to_tensor(x)
    return x * sigmoid(x)

def leaky_relu(x, negative_slope=0.2):
    x = dt.convert_to_tensor(x)
    return dt.where(dt.less(x, 0), x * negative_slope, x)

def hard_swish(x):
    x = dt.convert_to_tensor(x)
    return x * hard_sigmoid(x)

def elu(x, alpha=1.0):
    x = dt.convert_to_tensor(x)
    alpha = dt.convert_to_tensor(alpha)
    y = alpha * (dt.exp(x) - 1)
    return dt.where(dt.greater(x, 0), x, y)

def logsigmoid(x):
    x = dt.convert_to_tensor(x)
    return -softplus(-x)

def softmax(x, axis=-1):
    x = dt.convert_to_tensor(x)

    x_max = dt.max(x, axis=axis, keepdims=True)
    x_exp = dt.exp(x - x_max)
    x_sum = dt.sum(x_exp, axis=axis, keepdims=True)
    
    return x_exp / x_sum


def log_softmax(x, axis=-1):
    x = dt.convert_to_tensor(x)

    max_x = dt.max(x, axis=axis, keepdims=True)
    exp_x = dt.exp(x - max_x)
    logsumexp = dt.log(dt.sum(exp_x, axis=axis, keepdims=True))
    return x - max_x - logsumexp
