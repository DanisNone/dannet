import abc
from typing import Callable, ClassVar, Hashable

from dannet import lib
from dannet.lib import core, typing, utils
from dannet.lib.core import SymbolicTensor
from dannet.lib.core import SymbolicBuffer

from dannet.lib import dtypes
from dannet.lib.dtypes import DannetDtype


class Reduction(SymbolicTensor):
    _name: ClassVar[str]

    def __init__(self, x: SymbolicTensor, reduce_ndim: int, dtype: DannetDtype | None):
        self.x = x
        self._reduce_ndim = reduce_ndim

        self._shape = self.x.shape[:self.x.ndim-reduce_ndim]
        self._strides = core.default_strides(self.shape)
        self._offset = 0
        self._dtype = self.result_dtype(self.x.dtype, dtype)
        self._buffer = SymbolicBuffer(self)
        self._inner_size = self.x.size // self.size

    def inputs(self) -> list[SymbolicTensor]:
        return [self.x]

    def get_config(self) -> dict[str, Hashable]:
        return {"reduce_ndim": self._reduce_ndim}

    @abc.abstractmethod
    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        ...


class Sum(Reduction):
    _name = "sum"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            return dtype
        if dtypes.is_signed_dtype(dtype1) or dtypes.is_bool_dtype(dtype1):
            return dtypes.int64
        if dtypes.is_unsigned_dtype(dtype1):
            return dtypes.uint64
        return dtype1


class Mean(Reduction):
    _name = "mean"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            raise NotImplementedError()
        return dtypes.promote_to_inexact(dtype1)


class Prod(Reduction):
    _name = "prod"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            raise NotImplementedError()
        if dtypes.is_signed_dtype(dtype1) or dtypes.is_bool_dtype(dtype1):
            return dtypes.int64
        if dtypes.is_unsigned_dtype(dtype1):
            return dtypes.uint64
        return dtype1


class Min(Reduction):
    _name = "min"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            raise NotImplementedError()
        return dtype1


class Max(Reduction):
    _name = "max"

    def result_dtype(self, dtype1: DannetDtype, dtype: DannetDtype | None) -> DannetDtype:
        if dtype is not None:
            raise NotImplementedError()
        return dtype1


_reduction_func_type = Callable[
    [core.BaseTensor, typing.Axis, bool, DannetDtype | None],
    core.BaseTensor
]


def make_reduction(op_class: type[Reduction]) -> _reduction_func_type:
    def inner(
        x: core.BaseTensor, axis: typing.Axis,
        keepdims: bool = False,
        dtype: DannetDtype | None = None
    ) -> core.BaseTensor:
        x = core.to_symbolic(x)
        if axis is None:
            axis = range(x.ndim)
        axis = utils.normalize_axis_tuple(axis, x.ndim, "axis")
        keepdims = bool(keepdims)

        perm: list[int] = sorted(
            range(x.ndim), key=lambda i: (i in axis, abs(x.strides[i])))
        x = lib.as_strides.transpose(x, perm)

        out: core.BaseTensor = op_class(core.to_symbolic(x), len(axis), dtype)
        if keepdims:
            inv_perm = [perm.index(i) for i in range(len(perm))]
            out = lib.as_strides.expand_dims(out, range(out.ndim, x.ndim))
        else:
            dims = [i for i in range(x.ndim) if i not in axis]
            inv_perm = [perm.index(i) for i in dims]
        return lib.as_strides.transpose(out, inv_perm)
    inner.__name__ = op_class._name
    return inner


def sum(
    x: core.BaseTensor, axis: typing.Axis,
    keepdims: bool = False,
    dtype: DannetDtype | None = None,
    promote_integers: bool = True,
) -> core.BaseTensor:
    x = core.to_symbolic(x)
    if dtype is None:
        dtype = x.dtype
        if promote_integers:
            if dtypes.is_signed_dtype(x.dtype) or dtypes.is_bool_dtype(x.dtype):
                dtype = dtypes.int64
            if dtypes.is_unsigned_dtype(x.dtype):
                dtype = dtypes.uint64
    return _sum(x, axis, keepdims, dtype)


_sum = make_reduction(Sum)
mean = make_reduction(Mean)
prod = make_reduction(Prod)
min = make_reduction(Min)
max = make_reduction(Max)
