from __future__ import annotations

import dannet as dt
import numpy as np


def normalize_dtype(dtype: dt.typing.DTypeLike) -> str:
    if isinstance(dtype, np.dtype):
        dtype = dtype.name
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype).name
    elif dtype is bool or dtype == 'bool':
        dtype = bool_dtype
    elif dtype is int or dtype == 'int':
        dtype = int_dtype
    elif dtype is float or dtype == 'float':
        dtype = float_dtype
    elif isinstance(dtype, str):
        dtype = dtype.lower()
    else:
        raise TypeError(f'fail convert {dtype!r} to dtype')

    if dtype not in support:
        raise TypeError(f'unknown dtype: {dtype!r}')
    return dtype


def itemsize(dtype: dt.typing.DTypeLike) -> int:
    return np.dtype(normalize_dtype(dtype)).itemsize


def max_dtype(*dtypes: dt.typing.DTypeLike) -> str:
    dtypes = tuple(normalize_dtype(dtype) for dtype in dtypes)
    if len(dtypes) == 0:
        return bool_dtype
    return np.result_type(*dtypes).name


def is_bool_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return np.issubdtype(np.dtype(normalize_dtype(dtype)), np.bool_)


def is_integer_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return np.issubdtype(np.dtype(normalize_dtype(dtype)), np.integer)


def is_signed_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return np.issubdtype(np.dtype(normalize_dtype(dtype)), np.signedinteger)


def is_unsigned_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return np.issubdtype(np.dtype(normalize_dtype(dtype)), np.unsignedinteger)


def is_float_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return np.issubdtype(np.dtype(normalize_dtype(dtype)), np.floating)


def to_signed_dtype(dtype: dt.typing.DTypeLike) -> str:
    dtype = np.dtype(normalize_dtype(dtype))
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f'fail convert to signed: {dtype.name!r}')
    if np.issubdtype(dtype, np.signedinteger):
        return dtype.name
    return np.dtype(f'int{dtype.itemsize * 8}').name


def to_unsigned_dtype(dtype: dt.typing.DTypeLike) -> str:
    dtype = np.dtype(normalize_dtype(dtype))
    if not np.issubdtype(dtype, np.integer):
        raise TypeError(f'fail convert to unsigned: {dtype.name!r}')
    if np.issubdtype(dtype, np.unsignedinteger):
        return dtype.name
    return np.dtype(f'uint{dtype.itemsize * 8}').name


support = [
    'bool',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'uint16',
    'uint32',
    'uint64',

    'float16',
    'float32',
    'float64',
]


int_dtype = 'int32'
uint_dtype = 'uint32'
float_dtype = 'float32'
bool_dtype = 'bool'
