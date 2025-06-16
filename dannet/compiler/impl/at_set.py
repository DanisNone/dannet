from dannet.compiler.impl.utils import BuildInfo, build_program, get_shape, get_shape_info
from dannet.device import Device, DeviceEvent, DeviceKernel
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import at
from dannet.lib.dtypes import DannetDtype


def compile_copy(device: Device, dtypeA: DannetDtype) -> DeviceKernel:
    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
    )
    return build_program(device, "copy.cl", build_info).copy


def compile_set(device: Device, dtypeA: DannetDtype, dtypeB: DannetDtype) -> DeviceKernel:
    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB
    )
    return build_program(device, "at_set.cl", build_info).at_set


@register_impl(at.AtSet)
def at_set(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, at.AtSet)
    copy_kernel = compile_copy(device, node.dtype)
    set_kernel = compile_set(device, node.x.dtype, node.dtype)

    def inner(
        inputs: list[core.ConcreteTensor],
        output: core.ConcreteTensor
    ) -> DeviceEvent:
        x, values = inputs

        event1 = copy_kernel(
            node.size,
            None,
            x.buffer, output.buffer,
            get_shape_info(device, x),
            get_shape_info(device, output),
        )

        event2 = set_kernel(
            values.size,
            None,
            values.buffer, output.buffer,
            get_shape_info(device, values),
            get_shape_info(device, output),

            get_shape(device, node.start),
            get_shape(device, node.step),
            get_shape(device, core.default_strides(node.values.shape)),
            wait_for=event1
        )
        return event2
    return inner
