from __future__ import annotations

from typing import Hashable
import dannet
from dannet import lib
from dannet.lib.core import BaseTensor, SymbolicTensor, SymbolicBuffer
from dannet.lib.core import TensorLike, parse_getitem_key
from dannet.lib.core import _getitem_key_type


class AtSet(SymbolicTensor):
    def __init__(
        self,
        x: SymbolicTensor,
        values: SymbolicTensor,
        slices: tuple[tuple[int, int, int], ...]
    ):
        self.x = x
        self.values = values

        start, stop, step = [], [], []
        for s1, s2, s3 in slices:
            start.append(s1)
            stop.append(s2)
            step.append(s3)

        self.start = tuple(start)
        self.stop = tuple(stop)
        self.step = tuple(step)
        self._shape = self.x.shape
        self._strides = lib.core.default_strides(self._shape)
        self._offset = 0
        self._dtype = self.x._dtype

        self._buffer = SymbolicBuffer(self)

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x, self.values]

    def get_config(self) -> dict[str, Hashable]:
        return {"start": self.start, "stop": self.stop, "step": self.step}


class AtObject:
    def __init__(self, x: BaseTensor):
        self.x = x

    def __getitem__(self, key: _getitem_key_type) -> AtSliceObject:
        return AtSliceObject(self.x, key)


class AtSliceObject:
    def __init__(self, x: BaseTensor, key: _getitem_key_type):
        self.x = x
        if not isinstance(key, tuple):
            key = key,
        key = tuple(el for el in key if el is not None)
        slices, newaxes_positions, squeeze_axes = parse_getitem_key(x, key)
        slices += (slice(None, None, None), ) * (self.x.ndim - len(slices))

        assert len(slices) <= self.x.ndim
        assert newaxes_positions == []

        self.slices = slices
        self.squeeze_axes = squeeze_axes
        self.key = lib.core.normalize_slices(
            [(s.start, s.stop, s.step) for s in slices], x.shape, x.strides)[0]
        self.out_shape = self.x.__getitem__(key).shape

    def set(self, values: TensorLike) -> BaseTensor:
        values = dannet.broadcast_to(values, self.out_shape)
        values = dannet.expand_dims(values, self.squeeze_axes)
        values = values.astype(self.x.dtype)
        return dannet.core.at_set_jit(self.x, values, self.key)


def at_set(
    x: BaseTensor,
    values: BaseTensor,
    slices: tuple[tuple[int, int, int], ...]
) -> BaseTensor:
    x = lib.core.to_symbolic(x)
    values = lib.core.to_symbolic(values)

    return AtSet(x, values, slices)
