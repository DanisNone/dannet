from typing import Callable, TypeAlias
from dannet.lib.core import SymbolicTensor, ConcreteTensor
from dannet.device import Device, DeviceEvent


kernel_func_type: TypeAlias = Callable[[
    list[ConcreteTensor], ConcreteTensor], DeviceEvent | None]
compile_func_type: TypeAlias = Callable[
    [Device, SymbolicTensor],
    kernel_func_type
]

registered_impls: dict[type[SymbolicTensor], compile_func_type] = {}


def register_impl(
    op: type[SymbolicTensor]
) -> Callable[[compile_func_type], compile_func_type]:
    def inner(func: compile_func_type) -> compile_func_type:
        if op in registered_impls:
            raise ValueError(
                f"Implementation for {op.__name__} is already registered."
            )
        registered_impls[op] = func
        return func
    return inner


def compile_node(device: Device, node: SymbolicTensor) -> kernel_func_type:
    t = type(node)
    if t not in registered_impls:
        raise NotImplementedError(
            f'No implementation registered for {type(node)}'
        )
    func = registered_impls[t]
    return func(device, node)
