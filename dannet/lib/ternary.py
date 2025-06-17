import abc
from typing import Callable, ClassVar, Hashable
from dannet.lib import core
from dannet.lib.core import SymbolicTensor
from dannet.lib.core import SymbolicBuffer

from dannet.lib import dtypes
from dannet.lib.dtypes import DannetDtype


class Ternary(SymbolicTensor):
    _name: ClassVar[str]

    def __init__(
        self,
        x1: SymbolicTensor,
        x2: SymbolicTensor,
        x3: SymbolicTensor,
        dtype: DannetDtype | None
    ):
        core.require_equal_shape(self._name, x1, x2, x3)

        self.x1 = x1
        self.x2 = x2
        self.x3 = x3

        self._shape = self.x1.shape
        self._strides = core.default_strides(self.shape)
        self._offset = 0
        self._dtype = self.result_dtype(
            self.x1.dtype, self.x2.dtype, self.x3.dtype,
            dtype
        )
        self._buffer = SymbolicBuffer(self)

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x1, self.x2, self.x3]

    def get_config(self) -> dict[str, Hashable]:
        return {}

    @abc.abstractmethod
    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype3: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:
        ...


class Where(Ternary):
    _name = "where"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype3: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:
        if dtype is None:
            dtype = dtypes.promote_types(dtype2, dtype3)
        return dtype


_ternary_func_type = Callable[
    [core.BaseTensor, core.BaseTensor, core.BaseTensor, DannetDtype | None],
    core.BaseTensor
]


def make_ternary(op_class: type[Ternary]) -> _ternary_func_type:
    def inner(
        x1: core.BaseTensor,
        x2: core.BaseTensor,
        x3: core.BaseTensor, /,
        dtype: DannetDtype | None
    ) -> SymbolicTensor:
        x1 = core.to_symbolic(x1)
        x2 = core.to_symbolic(x2)
        x3 = core.to_symbolic(x3)
        return op_class(x1, x2, x3, dtype)
    inner.__name__ = op_class._name
    return inner


where = make_ternary(Where)
