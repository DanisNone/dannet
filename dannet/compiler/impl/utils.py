from __future__ import annotations


import math
from pathlib import Path
from typing import Sequence
import pyopencl as cl

import dannet as dt
from dannet.compiler.reg_impl import register_impl  # noqa: F401

dtype_map = {
    'bool': 'bool',
    'uint8': 'uchar',
    'uint16': 'ushort',
    'uint32': 'uint',
    'uint64': 'ulong',
    'int8': 'char',
    'int16': 'short',
    'int32': 'int',
    'int64': 'long',
    'float16': 'half',
    'float32': 'float',
    'float64': 'double',
}


def to_cl_dtype(dtype: str) -> str:
    if dtype not in dtype_map:
        raise ValueError(f'Unsupported dtype: {dtype}')
    return dtype_map[dtype]


def generate_static_array(name: str, values: Sequence[int]) -> str:
    values = list(values)
    values.append(0)

    array_str = ', '.join(str(v) for v in values)
    return f'__constant size_t {name}[{len(values)}] = {{{array_str}}};'


def generate_nodes_info(**kwargs: dt.core.TensorBase) -> list[str]:
    result = []
    for name, node in kwargs.items():
        result.append(f'#define dtype{name} {to_cl_dtype(node._dtype)}')
        result.append(generate_static_array(f'shape{name}', node._shape))
        result.append(generate_static_array(f'strides{name}', node._strides))
        result.append(f'#define offset{name} {node._buffer_offset}')
        result.append(f'#define ndim{name} {node.ndim}')
        result.append(f'#define size{name} {node.size}')
        result.append('')
    return result


def generate_defines(**kwargs) -> list[str]:
    res = []
    for key, value in kwargs.items():
        res.append(f'#define {key} {value}')
    return res


def generate_mode(mode: str):
    return f'#define {mode}'


_build_cache = {}


def build_kernel(device: dt.Device, name: str, headers: list[str] = []):
    key = (device, name, *headers)
    if key in _build_cache:
        return _build_cache[key]
    path = Path(__file__).parent.parent / 'kernels' / name

    with open(path, encoding='utf-8') as file:
        code = '\n'.join([*headers, file.read()])
    with open('last_compiled.cl', 'w') as file:
        print(f'//{name}\n'+code, file=file)
    _build_cache[key] = cl.Program(device.context, code).build()
    return _build_cache[key]


def default_strides(obj):
    if hasattr(obj, 'shape'):
        obj = obj.shape
    
    obj = dt.utils.normalize_shape(obj)
    return [math.prod(obj[i + 1:]) for i in range(len(obj))]
