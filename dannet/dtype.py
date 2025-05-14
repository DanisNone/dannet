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
    return _size_table[normalize_dtype(dtype)]

def max_dtype(*dtypes: dt.typing.DTypeLike) -> str:
    dtypes = tuple(normalize_dtype(dtype) for dtype in dtypes)
    common = set.intersection(*(_reachable_dict[name] for name in dtypes))

    for dtype in support:
        if dtype in common:
            return dtype

    raise TypeError('No common dtype found')



def is_bool_dtype(dtype):
    return normalize_dtype(dtype) == 'bool'

def is_integer_dtype(dtype):
    return normalize_dtype(dtype) in [
        'int8',
        'int16',
        'int32',
        'int64',
        'uint8',
        'uint16',
        'uint32',
        'uint64',
    ]

def signed_dtype(dtype):
    return normalize_dtype(dtype) in [
        'int8',
        'int16'
        'int32'
        'int64'
    ]

def is_unsigned_dtype(dtype):
    return normalize_dtype(dtype) in [
        'uint8',
        'uint16'
        'uint32'
        'uint64'
    ]

def is_float_dtype(dtype):
    return normalize_dtype(dtype) in [
        'float16',
        'float32',
        'float64',
    ]


def to_signed_dtype(dtype):
    dtype = normalize_dtype(dtype)
    if not is_integer_dtype(dtype):
        raise TypeError(f'fail convert to signed: {dtype!r}')
    return {
        'uint8': 'int8',
        'uint16': 'int16',
        'uint32': 'int32',
        'uint64': 'int64',
    }[dtype]

def to_unsigned_dtype(dtype):
    dtype = normalize_dtype(dtype)
    if not is_integer_dtype(dtype):
        raise TypeError(f'fail convert to unsigned: {dtype!r}')
    return {
        'int8': 'uint8',
        'int16': 'uint16',
        'int32': 'uint32',
        'int64': 'uint64',
    }[dtype]

def _reachable_from(dtype: str) -> set[str]:
    reached = {dtype}
    frontier = {dtype}
    while frontier:
        next_frontier = set()
        for t in frontier:
            for neighbor in graph.get(t, []):
                if neighbor not in reached:
                    reached.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return reached


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

graph = {
    'bool': ['uint8'],
    'uint8': ['int8', 'uint16'],
    'uint16': ['int16', 'uint32'],
    'uint32': ['int32', 'uint64'],
    'uint64': ['int64'],
    'int8': ['int16'],
    'int16': ['int32', 'float16'],
    'int32': ['int64', 'float32'],
    'int64': ['float64'],
    'float16': ['float32'],
    'float32': ['float64'],
    'float64': [],
}

_size_table = {
    'bool': 1,
    'uint8': 1,
    'uint16': 2,
    'uint32': 4,
    'uint64': 8,
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8,
    'float16': 2,
    'float32': 4,
    'float64': 8,
}

_reachable_dict = {dtype: _reachable_from(dtype) for dtype in support}


int_dtype = 'int32'
uint_dtype = 'uint32'
float_dtype = 'float32'
bool_dtype = 'bool'