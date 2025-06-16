import dannet as dt
from dannet.device import Device, DeviceEvent
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import binary
from dannet.lib.dtypes import DannetDtype
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_binary(
    device: Device,
    inputs: tuple[DannetDtype, DannetDtype],
    output: DannetDtype,
    op_name: str,
    cast: str = "CC->C",
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    inp, out = cast.split("->")
    assert len(inp) == 2
    assert len(out) == 1
    workA, workB, workC = inp[0], inp[1], out

    work_dtypes = {"A": dtypeA, "B": dtypeB, "C": dtypeC}

    op = (f"""
dt_$dtypeC$ operation(dt_$dtypeA$ x_inp, dt_$dtypeB$ y_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    dt_$workB$ y = dt_convert_$dtypeB$_to_$workB$(y_inp);
    return dt_{op_name}_$workC$(x, y);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
        workA=work_dtypes[workA],
        workB=work_dtypes[workB],
        workC=work_dtypes[workC],
    )
    build_info.add_header(op)
    return build_program(device, "binary.cl", build_info).binary


def make_binary(
    device: Device, node: core.SymbolicTensor,
    cast: str = "CC->C"
) -> kernel_func_type:
    assert isinstance(node, binary.Binary)
    kernel = compile_binary(
        device, (node.x1.dtype, node.x2.dtype), node.dtype, node._name, cast)

    def inner(
        inputs: list[core.ConcreteTensor],
        output: core.ConcreteTensor
    ) -> DeviceEvent:
        x1, x2 = inputs
        return kernel(
            node.size,
            None,
            x1.buffer, x2.buffer,
            output.buffer,

            get_shape_info(device, x1),
            get_shape_info(device, x2),
            get_shape_info(device, output)
        )
    return inner


register_impl(binary.Add)(make_binary)
register_impl(binary.Subtract)(make_binary)
register_impl(binary.Multiply)(make_binary)
register_impl(binary.Divide)(make_binary)
register_impl(binary.Arctan2)(make_binary)
