from typing import Hashable
from dannet.lib import core, dtypes
from dannet.lib.core import SymbolicBuffer, SymbolicTensor


class Matmul(SymbolicTensor):
    def __init__(self, x1: SymbolicTensor, x2: SymbolicTensor):
        if (
            x1.ndim != x2.ndim or
            x1.shape[:-2] != x2.shape[:-2] or
            x1.shape[-1] != x2.shape[-2]
        ):
            raise ValueError(""
                             "invalid shapes for matmul: "
                             f"{x1.shape=}, {x2.shape=}"
                             )

        self.x1 = x1
        self.x2 = x2

        self._shape = self.x1.shape[:-1] + (self.x2.shape[-1], )
        self._strides = core.default_strides(self.shape)
        self._offset = 0
        self._dtype = dtypes.promote_types(self.x1.dtype, self.x2.dtype)
        self._buffer = SymbolicBuffer(self)

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x1, self.x2]

    def get_config(self) -> dict[str, Hashable]:
        return {}


def matmul(x1: core.BaseTensor, x2: core.BaseTensor, /) -> SymbolicTensor:
    x1 = core.to_symbolic(x1)
    x2 = core.to_symbolic(x2)
    return Matmul(x1, x2)
