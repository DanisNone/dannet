from __future__ import annotations

import dannet as dt
import numpy as np


def normalize_dtype(dtype: dt.typing.DTypeLike) -> str:
    if isinstance(dtype, np.dtype):
        dtype = dtype.name
    elif isinstance(dtype, type) and issubclass(dtype, np.generic):
        dtype = np.dtype(dtype).name
    elif dtype is bool or dtype == 'bool':
        dtype = py_bool
    elif dtype is int or dtype == 'int':
        dtype = py_int
    elif dtype is float or dtype == 'float':
        dtype = py_float
    elif dtype is complex or dtype == 'complex':
        dtype = py_complex
    elif isinstance(dtype, str):
        dtype = dtype.lower()
    else:
        raise TypeError(f'fail convert {dtype!r} to dtype')

    if dtype not in support:
        raise TypeError(f'unknown dtype: {dtype!r}')
    return dtype


def itemsize(dtype: dt.typing.DTypeLike) -> int:
    return np.dtype(normalize_dtype(dtype)).itemsize


def is_bool_dtype(dtype: dt.typing.DTypeLike) -> bool:
    return normalize_dtype(dtype) == bool_


def is_integer_dtype(dtype: dt.typing.DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        int8, int16, int32, int64,
        uint8, uint16, uint32, uint64
    ]

def is_signed_dtype(dtype: dt.typing.DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        int8, int16, int32, int64
    ]

def is_unsigned_dtype(dtype: dt.typing.DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        uint8, uint16, uint32, uint64
    ]

def is_float_dtype(dtype: dt.typing.DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        float16, float32, float64
    ]


def is_complex_dtype(dtype: dt.typing.DTypeLike) -> bool:
    dtype = normalize_dtype(dtype)
    return dtype in [
        complex64, complex128
    ]

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


def real_part_of_complex_dtype(dtype: dt.typing.DTypeLike) -> str:
    if normalize_dtype(dtype) == 'complex32':
        return normalize_dtype('float16')
    dtype = np.dtype(normalize_dtype(dtype))
    if not np.issubdtype(dtype, np.complexfloating):
        raise TypeError(f'fail get real part: {dtype.name!r}')
    return normalize_dtype(f'float{dtype.itemsize * 8//2}')


def promote_dtypes(*dtypes: dt.typing.DTypeLike) -> str:
    norm_dtypes = sorted(
        (normalize_dtype(dtype) for dtype in dtypes),
        key=support.index, reverse=True
    )
    if len(norm_dtypes) == 0:
        return bool_

    result = norm_dtypes[0]
    for dtype in norm_dtypes[1:]:
        result = _promote_dtypes2(result, dtype)
    return result


def promote_to_float(dtype: dt.typing.DTypeLike):
    dtype = normalize_dtype(dtype)
    if is_float_dtype(dtype) or is_complex_dtype(dtype):
        return dtype
    bits = itemsize(dtype) * 8
    bits = max(bits, 32)
    return normalize_dtype(f'float{bits}')


def _promote_dtypes2(dtype1, dtype2):
    def from_info(rank, size):
        for key, v in _dtype_info.items():
            if (rank, size) == v:
                return key
        raise RuntimeError
    dtype1, dtype2 = sorted((dtype1, dtype2), key=support.index, reverse=True)

    type1, size1 = _dtype_info[dtype1]
    type2, size2 = _dtype_info[dtype2]

    if type2 == 'b':
        return dtype1
    if type1 == 'c':
        if type2 == 'c':
            return from_info('c', max(size1, size2))
        if type2 == 'f':
            return from_info('c', max(size1, size2*2))
        return dtype1
    if type1 == 'f':
        if type2 == 'f':
            return from_info('f', max(size1, size2))
        return from_info('f', size1)
    if type1 == 'i':
        if type2 == 'i':
            return from_info('i', max(size1, size2))
        if type2 == 'u':
            if size2 == 64:
                return from_info('f', size2)
            return from_info('i', max(size1, size2*2))
    if type1 == 'u':
        return from_info('u', max(size1, size2))
    raise RuntimeError('invalid dtypes')


support = [
    'bool',

    'uint8',
    'uint16',
    'uint32',
    'uint64',

    'int8',
    'int16',
    'int32',
    'int64',

    'float16',
    'float32',
    'float64',

    'complex64',
    'complex128',
]

_dtype_info = {
    'bool': ('b', 1),

    'int8': ('i', 8),
    'int16': ('i', 16),
    'int32': ('i', 32),
    'int64': ('i', 64),

    'uint8': ('u', 8),
    'uint16': ('u', 16),
    'uint32': ('u', 32),
    'uint64': ('u', 64),

    'float16': ('f', 16),
    'float32': ('f', 32),
    'float64': ('f', 64),

    'complex64': ('c', 64),
    'complex128': ('c', 128),
}

graph = {
    'bool': ['int8', 'uint8'],

    'int8': ['int16'],
    'int16': ['int32'],
    'int32': ['int64', 'float16'],
    'int64': ['float16'],

    'uint8': ['uint16', 'int16'],
    'uint16': ['uint32', 'int32'],
    'uint32': ['uint64', 'int64', 'float16'],
    'uint64': ['float64'],

    'float16': ['float32'],
    'float32': ['float64', 'complex64'],
    'float64': ['complex128'],

    'complex64': ['complex128'],
    'complex128': []
}



py_bool = 'bool'
py_int = 'int32'
py_float = 'float32'
py_complex = 'complex64'


bool_ = 'bool'
int8 = 'int8'
int16 = 'int16'
int32 = 'int32'
int64 = 'int64'
uint8 = 'uint8'
uint16 = 'uint16'
uint32 = 'uint32'
uint64 = 'uint64'

float16 = 'float16'
float32 = 'float32'
float64 = 'float64'

complex64 = 'complex64'
complex128 = 'complex128'