from __future__ import annotations
import dannet as dt


class _ConvND(dt.core.TensorBase):
    def __init__(self,
        rank: int,
        input: dt.typing.TensorLike,
        kernel: dt.typing.TensorLike, 
        strides: tuple[int, ...] | int = 1,
        padding: str ='valid'
    ):
        self.rank = int(rank)
        if self.rank != 2:
            raise NotImplementedError(f'{self.rank}-rank conv not implemented')
        
        self.input = dt.convert_to_tensor(input)
        self.kernel = dt.convert_to_tensor(kernel)
        self.conv_strides = normalize_strides(self.rank, strides)

        if self.input.ndim != self.rank + 2:
            raise ValueError(f'Input shape must have {self.rank + 2} dimensions, got {self.input.ndim}')
        if self.kernel.ndim != self.rank + 2:
            raise ValueError(f'Kernel shape must have {self.rank + 2} dimensions, got {self.kernel.ndim}')
        if len(self.conv_strides) != self.rank:
            raise ValueError(f'Strides must have {self.rank} dimensions, got {len(self.conv_strides)}')
        
        self.input = conv_pad(self.rank, self.input, self.kernel, self.conv_strides, padding)
        
        self._shape = conv_output_shape(self.input._shape, self.kernel._shape, self.conv_strides)
        self._dtype = dt.dtype.max_dtype(self.input._dtype, self.kernel._dtype, 'uint32')
        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.input, self.kernel]
    
    def compute_gradients(self, grad):
        batch, *x_shape, c1  = self.input._shape
        *k_shape, c1, c2 = self.kernel._shape

        grad_up_shape = [batch, *(x - k + 1 for x, k in zip(x_shape, k_shape)), c2]
        grad_up = _up_sample_zeros(grad, [1, *self.conv_strides, 1], grad_up_shape)

        axis = list(range(self.kernel.ndim))
        axis[-1], axis[-2] = axis[-2], axis[-1]
        
        k_transpose = dt.transpose(self.kernel, axis)
        k_flipped = dt.flip(k_transpose, range(self.rank))

        grad_x = dt.nnet.conv2d(grad_up, k_flipped, padding='full')


        axis = list(range(self.input.ndim))
        axis[0], axis[-1] = axis[-1], axis[0]
        x_transpose = dt.transpose(self.input, axis)

        axis = list(range(1, self.input.ndim - 1)) + [0, grad.ndim - 1]
        grad_transpose = dt.transpose(grad_up, axis)

        grad_k = dt.nnet.conv2d(x_transpose, grad_transpose)
        grad_k = dt.transpose(grad_k, axis)

        return [grad_x, grad_k]

class _DepthwiseConv2D(dt.core.TensorBase):
    def __init__(
        self,
        input: dt.typing.TensorLike,
        kernel: dt.typing.TensorLike,
        strides: tuple[int, int] | int = 1,
        padding: str = 'valid'
    ):
        self.input = dt.convert_to_tensor(input)
        self.kernel = dt.convert_to_tensor(kernel)

        self.conv_strides = normalize_strides(2, strides)

        if self.input.ndim != 4:
            raise ValueError(f'Input must have 4 dims, got {self.input.ndim}')
        
        if self.kernel.ndim == 3:
            self.kernel = dt.reshape(self.kernel, [*self.kernel._shape, 1])

        if self.kernel.ndim != 4:
            raise ValueError(f'Kernel must have 4 dims, got {self.kernel.ndim}')

        self.input = conv_pad(2, self.input, self.kernel, self.conv_strides, padding)
        self._shape = conv_output_shape_depthwise(self.input._shape, self.kernel._shape, self.conv_strides)

        self._dtype = dt.dtype.max_dtype(self.input._dtype, self.kernel._dtype, 'uint32')
        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.input, self.kernel]

    def compute_gradients(self, grad):
        raise NotImplementedError('gradient for depthwise2d conv not implemented')

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
        config['factors'] = self._upsample_size
        config['shape'] = self._shape
        return config


def normalize_strides(rank: int, strides: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(strides, int):
        strides = (strides, ) * rank
    strides = tuple(int(s) for s in strides)
    if len(strides) != rank:
        raise ValueError(f'Strides length {len(strides)} does not match rank {rank}')
    if min(strides) <= 0:
        raise ValueError('All stride values must be positive integers')
    return strides

def conv_output_shape(input_shape: tuple[int, ...], kernel_shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, ...]:
    B, *x, C1 = input_shape                
    *k, C1_, C2 = kernel_shape    

    if C1 != C1_:
        raise ValueError(f'Input channels ({C1}) and kernel channels ({C1_}) must match')

    result = []
    for inp, ker, stride in zip(x, k, strides):
        out = (inp - ker) // stride + 1
        if out <= 0:
            raise ValueError(f'Invalid output size computed: {(inp - ker)} // {stride} + 1 = {out}')
        result.append(out)
    return tuple([B, *result, C2])

def conv_output_shape_depthwise(input_shape: tuple[int, ...], kernel_shape: tuple[int, ...], strides: tuple[int, ...]) -> tuple[int, ...]:
    B, *x, C = input_shape
    *k, C_k, M = kernel_shape
    if C != C_k:
        raise ValueError(f'Input channels ({C}) and kernel channels ({C_k}) must match')
    result = []
    for inp, ker, stride in zip(x, k, strides):
        out = (inp - ker) // stride + 1
        if out <= 0:
            raise ValueError(f'Invalid output size: {(inp - ker)} // {stride} + 1 = {out}')
        result.append(out)
    return tuple([B, *result, C * M])

def conv_pad(rank: int, input: dt.core.TensorBase, kernel: dt.core.TensorBase, strides: tuple[int, ...], padding: str):
    if padding not in ('valid', 'same', 'full'):
            raise ValueError(f'invalid padding: {padding}')
    
    assert rank + 2 == input.ndim
    assert rank + 2 == kernel.ndim
    assert rank == len(strides)

    x_shape = input.shape[1:-1]
    k_shape = kernel.shape[:-2]
    paddings = []
    for x, k, s in zip(x_shape, k_shape, strides):
        if padding == 'valid':
            paddings.append((0, 0))
        
        elif padding == 'same':
            out = (x + s - 1) // s
            pad = max((out - 1) * s + k - x, 0)
            pad_left = pad // 2
            pad_right = pad - pad_left
            paddings.append((pad_left, pad_right))
        
        else:
            pad = k - 1
            paddings.append((pad, pad))
    paddings = [0, *paddings, 0]
    return dt.pad(input, paddings)


def _up_sample_zeros(x: dt.typing.TensorLike, factors: dt.typing.ShapeLike, shape: dt.typing.ShapeLike):
    x = dt.convert_to_tensor(x)
    y = _UpSampleZeros(x, factors, shape)

    if x.shape == y.shape:
        y = x
    return dt.core._node_prepare(y)


def conv2d(input: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides: tuple[int, int] | int = 1, padding: str = 'valid'):
    y = _ConvND(2, input, kernel, strides, padding)
    return dt.core._node_prepare(y)

def depthwise_conv2d(input: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides: tuple[int, int] | int = 1, padding: str = 'valid'):
    y = _DepthwiseConv2D(input, kernel, strides, padding)
    return dt.core._node_prepare(y)

__all__ = [
    'conv2d',
    'depthwise_conv2d'
]