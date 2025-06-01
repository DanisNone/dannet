from __future__ import annotations

from typing import Any

import dannet as dt
import numpy as np
import ml_dtypes


class DannetDtype(type):
    dtype: np.dtype

    def __hash__(self) -> int:
        return hash(self.dtype.type)

    def __eq__(self, other: Any) -> bool:
        return self is other or self.dtype.type == other

    def __ne__(self, other: Any) -> bool:
        return not (self == other)

    def __call__(self, x: Any) -> dt.core.Tensor:
        raise NotImplementedError()

    def __instancecheck__(self, instance: Any) -> bool:
        return isinstance(instance, self.dtype.type)


def _make_dtype(np_dtype: type) -> DannetDtype:
    dt_dtype = DannetDtype(
        np_dtype.__name__, (object, ),
        {"dtype": np.dtype(np_dtype)}
    )
    dt_dtype.__module__ = "dannet"
    return dt_dtype


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
