import abc
from typing import Callable, ClassVar, Hashable
from dannet.lib import core
from dannet.lib.core import SymbolicTensor
from dannet.lib.core import SymbolicBuffer

from dannet.lib import dtypes
from dannet.lib.dtypes import DannetDtype


class Binary(SymbolicTensor):
    _name: ClassVar[str]

    def __init__(self, x1: SymbolicTensor, x2: SymbolicTensor, dtype: DannetDtype | None):
        core.require_equal_shape(self._name, x1, x2)

        self.x1 = x1
        self.x2 = x2

        self._shape = self.x1.shape
        self._strides = core.default_strides(self.shape)
        self._offset = 0
        self._dtype = self.result_dtype(self.x1.dtype, self.x2.dtype, dtype)
        self._buffer = SymbolicBuffer(self)

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x1, self.x2]

    def get_config(self) -> dict[str, Hashable]:
        return {}

    @abc.abstractmethod
    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        ...


class Add(Binary):
    _name = "add"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        return dtype or dtypes.promote_types(dtype1, dtype2)


class Subtract(Binary):
    _name = "subtract"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        out = dtype or dtypes.promote_types(dtype1, dtype2)
        if out == dtypes.bool_:
            raise ValueError(
                "dannet boolean subtract, the `-` operator, is not supported",
                "use the bitwise_xor, the `^` operator, or the "
                "logical_xor functions instead."
            )
        return out


class Multiply(Binary):
    _name = "multiply"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        return dtype or dtypes.promote_types(dtype1, dtype2)


class Divide(Binary):
    _name = "divide"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        if dtype is None:
            dtype = dtypes.promote_types(dtype1, dtype2)
            dtype = dtypes.promote_to_inexact(dtype)
        elif not dtypes.is_inexact_dtype(dtype):
            raise TypeError(
                f"divide: Expected an inexact dtype, but got {dtype}")
        return dtype


class Arctan2(Binary):
    _name = "arctan2"

    def result_dtype(
        self,
        dtype1: DannetDtype,
        dtype2: DannetDtype,
        dtype: DannetDtype | None
    ) -> DannetDtype:

        if dtype is None:
            dtype = dtypes.promote_types(dtype1, dtype2)
            dtype = dtypes.promote_to_inexact(dtype)
        elif not dtypes.is_inexact_dtype(dtype):
            raise TypeError(
                f"arctan2: Expected an inexact dtype, but got {dtype}")
        return dtype


_binary_func_type = Callable[
    [core.BaseTensor, core.BaseTensor, DannetDtype | None],
    core.BaseTensor
]


def make_binary(op_class: type[Binary]) -> _binary_func_type:
    def inner(
        x1: core.BaseTensor,
        x2: core.BaseTensor, /,
        dtype: DannetDtype | None
    ) -> SymbolicTensor:
        x1 = core.to_symbolic(x1)
        x2 = core.to_symbolic(x2)
        return op_class(x1, x2, dtype)
    inner.__name__ = op_class._name
    return inner


add = make_binary(Add)
subtract = make_binary(Subtract)
multiply = make_binary(Multiply)
divide = make_binary(Divide)
arctan2 = make_binary(Arctan2)
