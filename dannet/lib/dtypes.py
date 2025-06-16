from __future__ import annotations

import functools
from typing import Any, TypeAlias

from dannet import lib
import numpy as np
import ml_dtypes


class DannetDtype(type):
    _instances: list[DannetDtype] = []
    dtype: np.dtype

    def __new__(cls, *args: Any, **kwargs: Any) -> DannetDtype:
        instance: DannetDtype = super().__new__(cls, *args, **kwargs)
        cls._instances.append(instance)
        return instance

    def __hash__(self) -> int:
        return hash(self.dtype.type)

    def __eq__(self, other: Any) -> bool:
        return self is other or self.dtype.type == other

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __call__(self, x: Any) -> "lib.core.BaseTensor":
        raise NotImplementedError()

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, self.dtype.type)


_PyScalar: TypeAlias = type[complex] | type[int] | type[float] | type[complex]
DTypeLike: TypeAlias = DannetDtype | np.dtype | str | _PyScalar


def _make_dtype(np_dtype: type) -> DannetDtype:
    dt_dtype = DannetDtype(
        np_dtype.__name__, (object, ),
        {"dtype": np.dtype(np_dtype)}
    )
    dt_dtype.__module__ = "dannet"
    return dt_dtype


def normalize_dtype(dtype: DTypeLike) -> DannetDtype:
    try:
        dtype = np.dtype(dtype)
    except TypeError as e:
        raise TypeError(e) from None

    for dt_dtype in DannetDtype._instances:
        if dtype == dt_dtype:
            return dt_dtype
    raise ValueError(
        f"dannet not support '{dtype!r}'"
    )


def is_bool_dtype(dtype: DTypeLike) -> bool:
    return normalize_dtype(dtype) == bool_


def is_integer_dtype(dtype: DTypeLike) -> bool:
    return is_signed_dtype(dtype) or is_unsigned_dtype(dtype)


def is_signed_dtype(dtype: DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        int8, int16, int32, int64
    ]


def is_unsigned_dtype(dtype: DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        uint8, uint16, uint32, uint64
    ]


def is_inexact_dtype(dtype: DTypeLike) -> bool:
    return is_float_dtype(dtype) or is_complex_dtype(dtype)


def is_float_dtype(dtype: DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        float16, bfloat16, float32, float64
    ]


def is_complex_dtype(dtype: DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        complex64, complex128
    ]


def real_part_of_complex(dtype: DTypeLike) -> DannetDtype:
    dtype_ = normalize_dtype(dtype)
    if not is_complex_dtype(dtype):
        raise TypeError("real_part_of_complex wait complex dtype")
    return {
        complex128: float64,
        complex64: float32
    }[dtype_]


def itemsize(dtype: DTypeLike) -> int:
    dtype = normalize_dtype(dtype)
    return np.dtype(dtype).itemsize


def promote_types(*dtypes: DTypeLike) -> DannetDtype:
    norm_dtypes = tuple(normalize_dtype(dtype) for dtype in dtypes)
    if len(norm_dtypes) == 0:
        return bool_

    result = norm_dtypes[0]
    for dtype in norm_dtypes[1:]:
        result = _promote_types2(result, dtype)
    return result


def promote_to_inexact(dtype: DTypeLike) -> DannetDtype:
    dtype_ = normalize_dtype(dtype)
    if is_float_dtype(dtype_) or is_complex_dtype(dtype_):
        return dtype_
    bits = itemsize(dtype_) * 8
    bits = max(bits, 32)
    return normalize_dtype(f'float{bits}')


@functools.cache
def _promote_types2(a: DannetDtype, b: DannetDtype) -> DannetDtype:
    p = _generate_promotions()
    CUB = p[a] & p[b]
    LUB = (CUB & {a, b}) or {c for c in CUB if CUB.issubset(p[c])}
    if len(LUB) != 1:
        raise RuntimeError(f"fail promote dtypes: {(a, b)=}")

    res = LUB.pop()
    if isinstance(res, DannetDtype):
        return res  # type: ignore
    return {
        int: int64,
        float: float64,
        complex: complex128
    }[res]


_promotions_type = dict[
    DannetDtype | type[int] | type[float] | type[complex],
    set[DannetDtype | type[int] | type[float] | type[complex]]
]


@functools.cache
def _generate_promotions() -> _promotions_type:
    integers = {
        dtype for dtype in DannetDtype._instances
        if is_integer_dtype(dtype)
    }
    floating = {
        dtype for dtype in DannetDtype._instances
        if is_float_dtype(dtype)
    }
    complexfloating = {
        dtype for dtype in DannetDtype._instances
        if is_complex_dtype(dtype)
    }

    result: _promotions_type = {}
    result[bool_] = {
        bool_, *integers, *floating, *complexfloating, int, float, complex
    }
    result[int] = {*integers, *floating, *complexfloating, int, float, complex}
    result[float] = {*floating, *complexfloating, float, complex}
    result[complex] = {*complexfloating, complex}

    for dtype1 in integers:
        result[dtype1] = {dtype1, *floating, *complexfloating, float, complex}
        for dtype2 in integers:
            s1 = is_signed_dtype(dtype1)
            s2 = is_signed_dtype(dtype2)
            if s1 and not s2:
                continue
            if (s1 != s2) and itemsize(dtype1) == itemsize(dtype2):
                continue
            if itemsize(dtype2) > itemsize(dtype1):
                result[dtype1].add(dtype2)

    for dtype1 in floating:
        result[dtype1] = {dtype1}

        for dtype2 in floating:
            if itemsize(dtype2) > itemsize(dtype1):
                result[dtype1].add(dtype2)
        for dtype2 in complexfloating:
            if itemsize(dtype2) >= itemsize(dtype1) * 2:
                result[dtype1].add(dtype2)

    for dtype1 in complexfloating:
        result[dtype1] = {dtype1}

        for dtype2 in complexfloating:
            if itemsize(dtype2) > itemsize(dtype1):
                result[dtype1].add(dtype2)
    return result


bool_ = _make_dtype(np.bool_)
uint8 = _make_dtype(np.uint8)
uint16 = _make_dtype(np.uint16)
uint32 = _make_dtype(np.uint32)
uint64 = _make_dtype(np.uint64)

int8 = _make_dtype(np.int8)
int16 = _make_dtype(np.int16)
int32 = _make_dtype(np.int32)
int64 = _make_dtype(np.int64)


float16 = _make_dtype(np.float16)
bfloat16 = _make_dtype(ml_dtypes.bfloat16)
float32 = _make_dtype(np.float32)
float64 = _make_dtype(np.float64)

complex64 = _make_dtype(np.complex64)
complex128 = _make_dtype(np.complex128)

finfo = np.finfo
__all__ = [
    "bool_",

    "uint8",
    "uint16",
    "uint32",
    "uint64",

    "int8",
    "int16",
    "int32",
    "int64",

    "bfloat16",
    "float16",
    "float32",
    "float64",

    "complex64",
    "complex128",

    "promote_types"
]
