import abc
from typing import Callable, ClassVar, Hashable
from dannet.lib import core
from dannet.lib.core import SymbolicTensor
from dannet.lib.core import SymbolicBuffer

from dannet.lib import dtypes
from dannet.lib.dtypes import DannetDtype


class Unary(SymbolicTensor):
    _name: ClassVar[str]

    def __init__(self, x: SymbolicTensor, dtype: DannetDtype | None):
        self.x = x

        self._shape = self.x.shape
        self._strides = core.default_strides(self.shape)
        self._offset = 0
        self._dtype = self.result_dtype(self.x.dtype, dtype)
        self._buffer = SymbolicBuffer(self)

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x]

    def get_config(self) -> dict[str, Hashable]:
        return {}

    @abc.abstractmethod
    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        ...


class UnaryInexact(Unary):
    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is None:
            dtype = dtypes.promote_to_inexact(dtype1)
        elif not dtypes.is_inexact_dtype(dtype):
            raise TypeError(
                f"{self._name}: Expected an inexact dtype, but got {dtype}")
        return dtype


class Negative(Unary):
    _name = "negative"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        dtype = dtype or dtype1
        if dtype == dtypes.bool_:
            raise ValueError(
                "The dannet boolean negative, the `-` operator, is not supported, "
                "use the `~` operator or the logical_not function instead."
            )
        return dtype


class Positive(Unary):
    _name = "positive"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        return dtype or dtype1


class Abs(Unary):
    _name = "abs"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            raise RuntimeError
        if dtypes.is_complex_dtype(dtype1):
            dtype1 = dtypes.real_part_of_complex(dtype1)
        return dtype1


class Square(Unary):
    _name = "square"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        return dtype or dtype1


class Sqrt(UnaryInexact):
    _name = "sqrt"


class Sign(Unary):
    _name = "sign"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        return dtype or dtype1


class Conjuagte(Unary):
    _name = "conjugate"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        dtype = dtype or dtype1
        if not dtypes.is_complex_dtype(dtype):
            raise ValueError("not use Conjugate for not complex tensors")
        return dtype


class Sin(UnaryInexact):
    _name = "sin"


class Cos(UnaryInexact):
    _name = "cos"


class Tan(UnaryInexact):
    _name = "tan"


class Sinh(UnaryInexact):
    _name = "sinh"


class Cosh(UnaryInexact):
    _name = "cosh"


class Tanh(UnaryInexact):
    _name = "tanh"


class Arcsin(UnaryInexact):
    _name = "arcsin"


class Arccos(UnaryInexact):
    _name = "arccos"


class Arctan(UnaryInexact):
    _name = "arctan"


class Arcsinh(UnaryInexact):
    _name = "arcsinh"


class Arccosh(UnaryInexact):
    _name = "arccosh"


class Arctanh(UnaryInexact):
    _name = "arctanh"


class Exp(UnaryInexact):
    _name = "exp"


class Exp2(UnaryInexact):
    _name = "exp2"


class Exp10(UnaryInexact):
    _name = "exp10"


class Expm1(UnaryInexact):
    _name = "expm1"


class Log(UnaryInexact):
    _name = "log"


class Log2(UnaryInexact):
    _name = "log2"


class Log10(UnaryInexact):
    _name = "log10"


class Log1p(UnaryInexact):
    _name = "log1p"


_unary_func_type = Callable[
    [core.BaseTensor, DannetDtype | None],
    core.BaseTensor
]


def make_unary(op_class: type[Unary]) -> _unary_func_type:
    def inner(
        x: core.BaseTensor, /,
        dtype: DannetDtype | None
    ) -> core.BaseTensor:
        x = core.to_symbolic(x)
        return op_class(x, dtype)
    inner.__name__ = op_class._name
    return inner


negative = make_unary(Negative)
positive = make_unary(Positive)
abs = make_unary(Abs)
square = make_unary(Square)
sqrt = make_unary(Sqrt)
sign = make_unary(Sign)
conjugate = make_unary(Conjuagte)

sin = make_unary(Sin)
cos = make_unary(Cos)
tan = make_unary(Tan)
sinh = make_unary(Sinh)
cosh = make_unary(Cosh)
tanh = make_unary(Tanh)

arcsin = make_unary(Arcsin)
arccos = make_unary(Arccos)
arctan = make_unary(Arctan)
arcsinh = make_unary(Arcsinh)
arccosh = make_unary(Arccosh)
arctanh = make_unary(Arctanh)

exp = make_unary(Exp)
exp2 = make_unary(Exp2)
exp10 = make_unary(Exp10)
expm1 = make_unary(Expm1)

log = make_unary(Log)
log2 = make_unary(Log2)
log10 = make_unary(Log10)
log1p = make_unary(Log1p)
