from __future__ import annotations
from typing import Any, Iterable

import numpy as np
import dannet as dt
import dannet.topsort as topsort


class function:
    _run_instance: function | None = None
    def __init__(self, func):
        self._func = func
        self._nodes: list[dt.core.TensorBase] = []

        self._cached_sign: dict[Any, tuple] = {}

    def __call__(self, *args, **kwargs):
        if function._run_instance is not None:
            raise NotImplementedError('Nested function calls are not supported')

        struct = get_struct((args, kwargs))

        placeholders: list[dt.core.Placeholder] = []
        inputs: list[np.ndarray] = []
        input_flatten = to_flatten((args, kwargs), struct)
        for i, obj in enumerate(input_flatten):
            if isinstance(obj, (np.ndarray, dt.core.Constant)):
                placeholder = dt.core.Placeholder(obj.shape, obj.dtype)
                input_flatten[i] = placeholder
                placeholders.append(placeholder)
                inputs.append(np.asarray(obj))
            elif isinstance(obj, dt.core.TensorBase):
                raise NotImplementedError('TensorBase inputs not supported yet')
        
        if struct in self._cached_sign:
            compiled, output_struct, output_indexes, output_template = self._cached_sign[struct]

            output_flatten = list(output_template)
            result_vals = compiled(inputs)
            for idx, val in zip(output_indexes, result_vals):
                output_flatten[idx] = val
            return to_struct(output_flatten, output_struct)

        if self._nodes:
            raise RuntimeError('Internal node buffer not empty before graph build')

        args_t, kwargs_t = to_struct(input_flatten, struct)
        function._run_instance = self
        output = self._func(*args_t, **kwargs_t)
        function._run_instance = None

        output_struct = get_struct(output)
        output_flatten = to_flatten(output, output_struct)
        outputs: list[dt.core.TensorBase] = []
        output_indexes: list[int] = []
        for i, out in enumerate(output_flatten):
            if isinstance(out, dt.core.TensorBase):
                outputs.append(out)
                output_indexes.append(i)

        compiled = dt.compiler.compile(placeholders, outputs, self._nodes, is_eager_mode=False)

        output_template: list[Any] = []
        for val in output_flatten:
            if isinstance(val, dt.core.TensorBase):
                output_template.append(None)
            else:
                output_template.append(val)

        self._cached_sign[struct] = (compiled, output_struct, output_indexes, output_template)

        self._nodes = []

        result_vals = compiled(inputs)
        for idx, val in zip(output_indexes, result_vals):
            output_flatten[idx] = val
        return to_struct(output_flatten, output_struct)

    @classmethod
    def _add_node(cls, node: dt.core.TensorBase):
        self = cls._run_instance
        if self is None:
            raise RuntimeError('Tensor operations must be used only in dt.function')

        for inp in node.inputs():
            if inp in self._nodes:
                continue
            if isinstance(inp, (dt.core.Placeholder, dt.core.Variable, dt.core.Constant)):
                self._nodes.append(inp)
            else:
                raise RuntimeError(f'node {node} have unknown input: {inp}')
        if node not in self._nodes:
            self._nodes.append(node)


def is_eager():
    return function._run_instance is None

def eval(x):
    x = dt.convert_to_tensor(x)
    if isinstance(x, dt.core.Constant):
        return x
    
    nodes = topsort.topological_sort([x])
    res = dt.compiler.compile([], [x], nodes[::-1], is_eager_mode=True)([])[0]
    return dt.constant(res)

def get_struct(obj) -> Any:
    if isinstance(obj, tuple):
        return ('tuple', tuple(get_struct(el) for el in obj))
    if isinstance(obj, list):
        return ('list', tuple(get_struct(el) for el in obj))
    if isinstance(obj, dict):
        return ('dict', tuple((get_struct(key), get_struct(value)) for key, value in obj.items()))
    if isinstance(obj, (np.ndarray, dt.core.TensorBase)):
        return ('ndarray', (obj.shape, str(obj.dtype)))
    if isinstance(obj, (bool, int, float, str)):
        return (type(obj).__name__, obj)
    return ('unknown', id(obj))


def to_flatten(obj, struct) -> list:
    type_, args = struct

    if type_ == 'tuple':
        assert isinstance(obj, tuple)
        assert len(obj) == len(args)
        return join_list([to_flatten(el, sub) for el, sub in zip(obj, args)])
    if type_ == 'list':
        assert isinstance(obj, list)
        assert len(obj) == len(args)
        return join_list([to_flatten(el, sub) for el, sub in zip(obj, args)])
    if type_ == 'dict':
        assert isinstance(obj, dict)
        assert len(obj) == len(args)

        res = []
        for (k_el, v_el), (k_sub, v_sub) in zip(obj.items(), args):
            res.extend(to_flatten(k_el, k_sub))
            res.extend(to_flatten(v_el, v_sub))
        return res
    if type_ == 'ndarray':
        assert isinstance(obj, (np.ndarray, dt.core.TensorBase))
        assert obj.shape == args[0]
        assert obj.dtype == args[1]

        return [obj]
    if type_ == 'bool':
        assert isinstance(obj, bool)
        assert obj == args

        return [obj]
    if type_ == 'int':
        assert isinstance(obj, int)
        assert obj == args

        return [obj]
    if type_ == 'float':
        assert isinstance(obj, float)
        assert obj == args

        return [obj]
    if type_ == 'str':
        assert isinstance(obj, str)
        assert obj == args

        return [obj]
    if type_ == 'unknown':
        assert id(obj) == args
        return [obj]
    
    raise TypeError(f'Uknown type: {type_!r}')
    
def to_struct(data: list, struct) -> Any:
    def _rebuild(iterator, struct):
        type_, args = struct

        if type_ == 'tuple':
            return tuple(_rebuild(iterator, sub) for sub in args)
        if type_ == 'list':
            return [_rebuild(iterator, sub) for sub in args]
        if type_ == 'dict':
            return {
                _rebuild(iterator, k): _rebuild(iterator, v)
                for k, v in args
            }
        if type_ == 'ndarray':
            value = next(iterator)
            assert isinstance(value, (np.ndarray, dt.core.TensorBase))
            assert value.shape == args[0]
            assert str(value.dtype) == args[1]
            return value
        if type_ == 'bool':
            value = next(iterator)
            assert isinstance(value, bool)
            assert value == args
            return value
        if type_ == 'int':
            value = next(iterator)
            assert isinstance(value, int)
            assert value == args
            return value
        if type_ == 'float':
            value = next(iterator)
            assert isinstance(value, float)
            assert value == args
            return value
        if type_ == 'str':
            value = next(iterator)
            assert isinstance(value, str)
            assert value == args
            return value
        if type_ == 'unknown':
            value = next(iterator)
            assert id(value) == args
            return value
    
        raise TypeError(f'Unknown type: {type_!r}')

    return _rebuild(iter(data), struct)

def join_list(lists: Iterable[list]):
    res = []
    for list in lists:
        res.extend(list)
    return res