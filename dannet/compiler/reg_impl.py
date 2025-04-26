from __future__ import annotations

from typing import Callable
import dannet as dt


impl_ops: dict[type[dt.core.TensorBase], Callable] = {}


def register_impl(op: type[dt.core.TensorBase]):
    def decorator(func: Callable):
        impl_ops[op] = func
        return func

    return decorator


def compile_node(device: dt.Device, node: dt.core.TensorBase, input_buffers, output_buffer):
    t = type(node)
    if t not in impl_ops:
        raise NotImplementedError(f"No implementation registered for {type(node)}")
    return impl_ops[t](device, node, input_buffers, output_buffer)
