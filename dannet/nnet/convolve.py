from __future__ import annotations
import dannet as dt


class _ConvND(dt.core.TensorBase):
    def __init__(self, x, kernel, stride):
        self.x = dt.convert_to_tensor(x)
        self.kernel = dt.convert_to_tensor(kernel)

        if not self.x._is_default_strides():
            self.x = dt.copy(self.x)
        if not self.kernel._is_default_strides():
            self.kernel = dt.copy(self.kernel)
        
        self.stride = stride

        self._shape = self._compute_output_shape()
        self._dtype = dt.dtype.max_dtype(self.x._dtype, self.kernel._dtype)

        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()

    def _compute_output_shape(self):
        raise NotImplementedError

    def inputs(self):
        return [self.x, self.kernel]


class _Conv1D(_ConvND):
    def _compute_output_shape(self):
        B, W, C_in = self.x._shape  
        KW, C_in_, C_out = self.kernel._shape
        
        if C_in != C_in_:
            raise ValueError(f'Incomplete shapes: {self.x._shape}, {self.kernel._shape}')

        SW = self.stride
        
        W_out = (W - KW) // SW + 1
        return (B, W_out, C_out)
    
    def compute_gradients(self, grad):
        raise NotImplementedError('Backprop for Conv1D not yet implemented.')


class _Conv2D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, C_in = self.x._shape                
        KW, KH, C_in_, C_out = self.kernel._shape    

        if C_in != C_in_:
            raise ValueError(f'Incompatible shapes: {self.x._shape}, {self.kernel._shape}')

        SW, SH = self.stride                         

        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        return (B, W_out, H_out, C_out)

    def compute_gradients(self, grad):
        batch, w, h, c1  = self.x._shape
        kw, kh, c1, c2 = self.kernel._shape

        grad = _up_sample_zeros(grad, [1, *self.stride, 1], (batch, w - kw + 1, h - kh + 1, c2))

        k = dt.transpose(self.kernel, (0, 1, 3, 2))
        k = dt.flip(k, (0, 1))

        grad_x = dt.nnet.conv2d(grad, k, padding='full')


        x = dt.transpose(self.x, (3, 1, 2, 0))
        g = dt.transpose(grad, (1, 2, 0, 3))

        grad_k = dt.nnet.conv2d(x, g)
        grad_k = dt.transpose(grad_k, (1, 2, 0, 3))

        assert grad_x.shape == self.x.shape, (grad_x.shape, self.x.shape)
        assert grad_k.shape == self.kernel.shape, (grad_k.shape, self.kernel.shape)
        
        return [grad_x, grad_k]


class _Conv3D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, D, C_in = self.x._shape
        KW, KH, KD, C_in_, C_out = self.kernel._shape
        
        if C_in != C_in_:
            raise ValueError(f'Incompatible shapes: {self.x._shape}, {self.kernel._shape}')

        SW, SH, SD = self.stride
        
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        D_out = (D - KD) // SD + 1
        return (B, W_out, H_out, D_out, C_out)

    def compute_gradients(self, grad):
        raise NotImplementedError('Backprop for Conv3D not yet implemented.')

class _DepthwiseConv1D(_ConvND):
    def _compute_output_shape(self):
        B, W, C = self.x._shape
        KW, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f'Incompatible shapes: {self.x._shape}, {self.kernel._shape}')
        SW = self.stride
        W_out = (W - KW) // SW + 1
        return (B, W_out, C)

class _DepthwiseConv2D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, C = self.x._shape
        KW, KH, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f'Incompatible shapes: {self.x._shape}, {self.kernel._shape}')
        SW, SH = self.stride
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        return (B, W_out, H_out, C)

class _DepthwiseConv3D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, D, C = self.x._shape
        KW, KH, KD, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f'Incompatible shapes: {self.x._shape}, {self.kernel._shape}')
        SW, SH, SD = self.stride
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        D_out = (D - KD) // SD + 1
        return (B, W_out, H_out, D_out, C)

class _UpSampleZeros(dt.core.TensorBase):
    def __init__(self, x, factors: dt.typing.ShapeLike, shape: dt.typing.ShapeLike):
        self.x = dt.convert_to_tensor(x)
        factors = dt.utils.normalize_shape(factors)
        shape = dt.utils.normalize_shape(shape)

        if len(factors) != self.x.ndim:
            raise ValueError('Factors must have the same ndim as input')
        if len(shape) != self.x.ndim:
            raise ValueError('Shape must have the same ndim as input')

        if min(factors) <= 0:
            raise ValueError('Upsample factors must be > 0')

        expected_shape = [dim * f for dim, f in zip(self.x._shape, factors)]
        for exp, shp in zip(expected_shape, shape):
            if exp < shp:
                raise ValueError(f'Cannot upsample to smaller shape: expected at least {expected_shape}, got {shape}')
        
        self._upsample_size = factors
        self._shape = shape
        self._dtype = self.x.dtype

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]
    
    def compute_gradients(self, grad):
        slices = [(None, None, s) for s in self._upsample_size]
        return [dt.slice(grad, slices)]
    
    def get_config(self):
        config = super(_UpSampleZeros, self).get_config()
        config["factors"] = self._upsample_size
        config["shape"] = self._shape
        return config


def _up_sample_zeros(x: dt.typing.TensorLike, factors: dt.typing.ShapeLike, shape: dt.typing.ShapeLike):
    x = dt.convert_to_tensor(x)
    y = _UpSampleZeros(x, factors, shape)

    if x.shape == y.shape:
        y = x
    return dt.core._node_prepare(y)

def _pad(x_shape, kernel_shape, stride, padding):
    if padding not in ('valid', 'same', 'full'):
            raise ValueError(f'invalid padding: {padding}')
    x_shape = dt.utils.normalize_shape(x_shape)
    kernel_shape = dt.utils.normalize_shape(kernel_shape)
    stride = dt.utils.normalize_shape(stride)
    
    assert len(x_shape) == len(kernel_shape) == len(stride)
    paddings = []
    for i in range(len(x_shape)):
        x_dim, k_dim, s = x_shape[i], kernel_shape[i], stride[i]

        if padding == 'valid':
            paddings.append((0, 0))
        elif padding == 'same':
            y_dim = (x_dim + s - 1) // s
            pad = max((y_dim - 1) * s + k_dim - x_dim, 0)
            pad_left = pad // 2
            pad_right = pad - pad_left
            paddings.append((pad_left, pad_right))
        else:
            pad = (k_dim - 1)
            paddings.append((pad, pad))
    return paddings



def _make_conv(name: str, class_: type[_ConvND]):
    def inner(x: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides=None, padding='valid'):
        x = dt.convert_to_tensor(x)
        kernel = dt.convert_to_tensor(kernel)
        if strides is None:
            strides = [1] * (x.ndim - 2)
        
        paddings = _pad(x.shape[1:-1], kernel.shape[:-2], strides,padding)
        paddings = [0, *paddings, 0]

        x = dt.pad(x, paddings)
        z = class_(x, kernel, strides)
        return dt.core._node_prepare(z)
    inner.__name__ = name
    return inner

def _make_depthwise_conv(name: str, class_: type[_ConvND]):
    def inner(x: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides=None, padding='valid'):
        x = dt.convert_to_tensor(x)
        kernel = dt.convert_to_tensor(kernel)

        if padding not in ('same', 'valid'):
            raise ValueError(f'invalid padding: {padding}')
        if strides is None:
            strides = [1] * (x.ndim - 2)

        if padding == 'same':
            _, *S, _ = x.shape
            *K, _ = kernel.shape
            paddings = []
            for dim, k, s in zip(S, K, strides):
                out_dim = (dim + s - 1) // s
                pad_needed = max((out_dim - 1) * s + k - dim, 0)
                pad_before = pad_needed // 2
                pad_after = pad_needed - pad_before
                paddings.append((pad_before, pad_after))
            paddings = [0, *paddings, 0]
            x = dt.pad(x, paddings)

        z = class_(x, kernel, strides)
        return dt.core._node_prepare(z)

    inner.__name__ = name
    return inner

conv1d = _make_conv('conv1d', _Conv1D)
conv2d = _make_conv('conv2d', _Conv2D)
conv3d = _make_conv('conv3d', _Conv3D)

depthwise_conv1d = _make_depthwise_conv('depthwise_conv1d', _DepthwiseConv1D)
depthwise_conv2d = _make_depthwise_conv('depthwise_conv2d', _DepthwiseConv2D)
depthwise_conv3d = _make_depthwise_conv('depthwise_conv3d', _DepthwiseConv3D)

__all__ = [
    'conv1d',
    'conv2d',
    'conv3d',
    'depthwise_conv1d',
    'depthwise_conv2d',
    'depthwise_conv3d'
]