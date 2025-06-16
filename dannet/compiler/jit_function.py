from typing import Any, Callable, Hashable
import numpy as np
import tree

from dannet.device import current_device
from dannet.lib import core
from dannet.lib.core import SymbolicTensor
from dannet.compiler.compiler import compile


_cache_item_type = tuple[
    compile,
    list[core.Placeholder],
    Any,
    list[SymbolicTensor]
]


class jit:
    _is_runned: bool = False

    def __init__(self, func: Callable[..., Any]):
        self._func = func
        self._cache: dict[Hashable, _cache_item_type] = {}

    def _make_cache_key(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
        input_info = []
        for arg in tree.flatten((args, kwargs)):
            if isinstance(arg, (np.ndarray, core.ConcreteTensor)):
                if isinstance(arg, np.ndarray):
                    arg = core.ConcreteTensor.from_ndarray(arg)
                input_info.append((
                    arg.shape,
                    arg.strides,
                    arg.dtype,
                    arg.buffer_offset
                ))
            else:
                input_info.append(arg)

        return tuple(input_info)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if jit._is_runned:
            return self._func(*args, **kwargs)

        device = current_device()
        cache_key = self._make_cache_key(args, kwargs)

        if cache_key not in self._cache:
            flatten_input = tree.flatten((args, kwargs))

            inputs: list[Any] = []
            placeholders: list[core.Placeholder] = []
            values: list[core.ConcreteTensor] = []
            for arg in flatten_input:
                if isinstance(arg, np.ndarray):
                    arg = core.ConcreteTensor.from_ndarray(arg)

                if isinstance(arg, core.ConcreteTensor):
                    values.append(arg)
                    arg = core.Placeholder(arg)
                    placeholders.append(arg)
                elif isinstance(arg, core.BaseTensor):
                    raise TypeError(
                        "JIT compilation requires concrete tensors as inputs. "
                        f"Got tensor of type {type(arg)!r} instead. "
                        "Please provide actual values, not symbolic tensors."
                    )
                inputs.append(arg)

            args_t, kwargs_t = tree.unflatten_as(
                (args, kwargs), inputs  # type: ignore
            )

            jit._is_runned = True
            try:
                output = self._func(*args_t, **kwargs_t)
            finally:
                jit._is_runned = False
            output_flatten = tree.flatten(output)

            output_symbolic = [out for out in output_flatten if isinstance(
                out, core.SymbolicTensor)]
            compiled = compile(device, placeholders, output_symbolic)

            self._cache[cache_key] = (
                compiled, placeholders, output, output_symbolic)

        compiled, placeholders, output, output_symbolic = self._cache[cache_key]

        values = []
        for arg in tree.flatten((args, kwargs)):
            if isinstance(arg, np.ndarray):
                arg = core.ConcreteTensor.from_ndarray(arg)
            if isinstance(arg, core.ConcreteTensor):
                values.append(arg)

        output_values = compiled(values)

        result = []
        output_flatten = tree.flatten(output)
        output_symbolic_ptr = 0
        for item in output_flatten:
            if isinstance(item, core.SymbolicTensor):
                result.append(output_values[output_symbolic_ptr])
                output_symbolic_ptr += 1
            else:
                result.append(item)
        return tree.unflatten_as(output, result)
