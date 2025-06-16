import dannet as dt
from dannet.device import Device, DeviceEvent
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import tensor_contractions
from dannet.lib.dtypes import DannetDtype
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_matmul(
    device: Device,
    inputs: tuple[DannetDtype, DannetDtype],
    output: DannetDtype,
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
    )
    return build_program(device, "matmul.cl", build_info).matmul


@register_impl(tensor_contractions.Matmul)
def make_matmul(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, tensor_contractions.Matmul)
    kernel = compile_matmul(device, (node.x1.dtype, node.x2.dtype), node.dtype)
    M, K = node.shape[-2:]
    batch = node.size // (M * K)

    def inner(
        inputs: list[core.ConcreteTensor],
        output: core.ConcreteTensor
    ) -> DeviceEvent:
        x1, x2 = inputs
        return kernel(
            (batch, M, K),
            None,
            x1.buffer, x2.buffer,
            output.buffer,

            get_shape_info(device, x1),
            get_shape_info(device, x2),
            get_shape_info(device, output)
        )
    return inner
