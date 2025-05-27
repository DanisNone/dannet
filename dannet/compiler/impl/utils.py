from __future__ import annotations


import functools
import math
from pathlib import Path
import re
import pyopencl as cl

import dannet as dt
from dannet.compiler.reg_impl import register_impl  # noqa: F401


class Generator:
    def __init__(self):
        self._defines: list[str] = []
        self._lines: list[str] = []
        self._replaces: dict[str, str] = {}

    def static_array(self, name: str, values: tuple[int, ...] | list[int]):
        vals = list(values) + [0]
        array_str = ', '.join(str(v) for v in vals)
        line = f'__constant size_t {name}[{len(vals)}] = {{{array_str}}};'
        self._lines.append(line)

    def nodes_info(self, **kwargs: dt.core.TensorBase):
        for name, node in kwargs.items():
            self.static_array(f'shape{name}', node._shape)
            self.static_array(f'strides{name}', node._strides)

            self._defines.append(f'#define offset{name} {node._buffer_offset}')
            self._defines.append(f'#define ndim{name} {node.ndim}')
            self._defines.append(f'#define size{name} {node.size}')

            self._replaces[f'dtype{name}'] = node._dtype

    def dtype_names(self, **kwargs: str):
        for name, dtype in kwargs.items():
            dtype = dt.dtype.normalize_dtype(dtype)
            self._replaces[name] = dtype

    def defines(self, **kwargs):
        for key, value in kwargs.items():
            self._defines.append(f'#define {key} {value}')

    def mode(self, mode_name: str):
        self._defines.append(f'#define {mode_name}')

    def line(self, line: str):
        self._lines.append(line)

    def get_all(self) -> str:
        parts = ['#include "dtypes/core.cl"']
        parts.extend(self._defines)
        parts.extend(self._lines)

        return '\n'.join(parts)

    def apply(self, code: str) -> str:
        result = '\n'.join((self.get_all(), code))
        for name, dtype in self._replaces.items():
            result = re.sub(f'\\b{name}\\b', f'dt_{dtype}', result)
            result = re.sub(name, dtype, result)

        return result

    def __eq__(self, other):
        if not isinstance(other, Generator):
            return False
        return (
            self._defines == other._defines and
            self._lines == other._lines and
            self._replaces == other._replaces
        )

    def __hash__(self):
        return hash(self.apply(''))


@functools.cache
def build_kernel(device: dt.Device, name: str, gen: Generator):
    root = Path(__file__).parent.parent
    path = root / 'kernels' / name

    with open(path, encoding='utf-8') as file:
        code = gen.apply(file.read())
    with open('last_compiled.cl', 'w') as file:
        print(f'//{name}\n'+code, file=file)
    options = [f'-I {root}']

    program = cl.Program(device.context, code).build(options)
    return program


def default_strides(obj):
    if hasattr(obj, 'shape'):
        obj = obj.shape
    obj = dt.utils.normalize_shape(obj)
    return [math.prod(obj[i + 1:]) for i in range(len(obj))]
