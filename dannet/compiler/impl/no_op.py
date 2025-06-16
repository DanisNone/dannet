from dannet.compiler.register import register_impl, kernel_func_type
from dannet.device import Device
from dannet.lib.core import ConcreteTensor, SymbolicTensor, Constant, Placeholder

from dannet.lib.as_strides import AsStrides


@register_impl(AsStrides)
def asstrides(device: Device, node: SymbolicTensor) -> kernel_func_type:
    def kernel(inputs: list[ConcreteTensor], output: ConcreteTensor) -> None:
        assert len(inputs) == 1
        assert inputs[0].buffer is output.buffer
        return None
    return kernel


@register_impl(Placeholder)
@register_impl(Constant)
def no_op(device: Device, node: SymbolicTensor) -> kernel_func_type:
    def kernel(inputs: list[ConcreteTensor], output: ConcreteTensor) -> None:
        assert len(inputs) == 0
        return None
    return kernel
