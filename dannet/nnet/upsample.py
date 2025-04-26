import dannet as dt

class _UpSample(dt.core.TensorBase):
    def __init__(self, x, scale):
        self.x = dt.convert_to_tensor(x)

        if isinstance(scale, int):
            self.scale = (scale,) * self.x.ndim
        else:
            self.scale = tuple(scale)
        if len(self.scale) != self.x.ndim:
            raise ValueError(f'Scale {self.scale} must match tensor ndim {self.x.ndim}')

        self._shape = tuple(s * sc for s, sc in zip(self.x.shape, self.scale))
        self._dtype = self.x.dtype

        self._strides = self._default_strides()
        self._buffer = dt.core.Buffer(self)
        self._buffer_offset = 0

    def inputs(self):
        return [self.x]

    def compute_gradients(self, grad):
        raise NotImplementedError()
    
    def get_config(self):
        cfg = super(_UpSample, self).get_config()
        cfg['scale'] = self.scale
        return cfg


def upsample(x, scale):
    x = dt.convert_to_tensor(x)
    t = _UpSample(x, scale)
    return dt.core._node_prepare(t)
