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

    def compute_gradients(self, grad):
        raise NotImplementedError("Backprop for ConvND not yet implemented.")


class _Conv1D(_ConvND):
    def _compute_output_shape(self):
        B, W, C_in = self.x._shape  
        KW, C_in_, C_out = self.kernel._shape
        
        if C_in != C_in_:
            raise ValueError(f"Incomplete shapes: {self.x._shape}, {self.kernel._shape}")

        SW = self.stride
        
        W_out = (W - KW) // SW + 1
        return (B, W_out, C_out)
    
    def compute_gradients(self, grad):
        raise NotImplementedError("Backprop for Conv1D not yet implemented.")


class _Conv2D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, C_in = self.x._shape                
        KW, KH, C_in_, C_out = self.kernel._shape    

        if C_in != C_in_:
            raise ValueError(f"Incompatible shapes: {self.x._shape}, {self.kernel._shape}")

        SW, SH = self.stride                         

        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        return (B, W_out, H_out, C_out)

    def compute_gradients(self, grad):
        raise NotImplementedError("Backprop for Conv2D not yet implemented.")

class _Conv3D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, D, C_in = self.x._shape
        KW, KH, KD, C_in_, C_out = self.kernel._shape
        
        if C_in != C_in_:
            raise ValueError(f"Incompatible shapes: {self.x._shape}, {self.kernel._shape}")

        SW, SH, SD = self.stride
        
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        D_out = (D - KD) // SD + 1
        return (B, W_out, H_out, D_out, C_out)

    def compute_gradients(self, grad):
        raise NotImplementedError("Backprop for Conv3D not yet implemented.")

class _DepthwiseConv1D(_ConvND):
    def _compute_output_shape(self):
        B, W, C = self.x._shape
        KW, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f"Incompatible shapes: {self.x._shape}, {self.kernel._shape}")
        SW = self.stride
        W_out = (W - KW) // SW + 1
        return (B, W_out, C)

class _DepthwiseConv2D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, C = self.x._shape
        KW, KH, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f"Incompatible shapes: {self.x._shape}, {self.kernel._shape}")
        SW, SH = self.stride
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        return (B, W_out, H_out, C)

class _DepthwiseConv3D(_ConvND):
    def _compute_output_shape(self):
        B, W, H, D, C = self.x._shape
        KW, KH, KD, C_ = self.kernel._shape
        if C != C_:
            raise ValueError(f"Incompatible shapes: {self.x._shape}, {self.kernel._shape}")
        SW, SH, SD = self.stride
        W_out = (W - KW) // SW + 1
        H_out = (H - KH) // SH + 1
        D_out = (D - KD) // SD + 1
        return (B, W_out, H_out, D_out, C)


def _make_conv(name: str, class_: type[_ConvND]):
    def inner(x: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides=None, padding="valid"):
        x = dt.convert_to_tensor(x)
        kernel = dt.convert_to_tensor(kernel)
        
        if padding not in ("same", "valid"):
            raise ValueError(f"invalid padding: {padding}")
        if strides is None:
            strides = [1] * (x.ndim - 2)
        
        if padding == "same":
            _, *S, _ = x.shape
            *K, _, _ = kernel.shape

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

def _make_depthwise_conv(name: str, class_: type[_ConvND]):
    def inner(x: dt.typing.TensorLike, kernel: dt.typing.TensorLike, strides=None, padding="valid"):
        x = dt.convert_to_tensor(x)
        kernel = dt.convert_to_tensor(kernel)

        if padding not in ("same", "valid"):
            raise ValueError(f"invalid padding: {padding}")
        if strides is None:
            strides = [1] * (x.ndim - 2)

        if padding == "same":
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

depthwise_conv1d = _make_depthwise_conv("depthwise_conv1d", _DepthwiseConv1D)
depthwise_conv2d = _make_depthwise_conv("depthwise_conv2d", _DepthwiseConv2D)
depthwise_conv3d = _make_depthwise_conv("depthwise_conv3d", _DepthwiseConv3D)

__all__ = [
    "conv1d",
    "conv2d",
    "conv3d",
    "depthwise_conv1d",
    "depthwise_conv2d",
    "depthwise_conv3d"
]