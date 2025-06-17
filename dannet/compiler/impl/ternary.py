from functools import partial
import dannet as dt
from dannet.device import Device, DeviceEvent
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import ternary
from dannet.lib.dtypes import DannetDtype
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_ternary(
    device: Device,
    inputs: tuple[DannetDtype, DannetDtype, DannetDtype],
    output: DannetDtype,
    op_name: str,
    cast: str = "DDD->D",
    custom_operation: str | None = None
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB, dtypeC = inputs
    dtypeD = output

    inp, out = cast.split("->")
    assert len(inp) == 3
    assert len(out) == 1
    workA, workB, workC, workD = inp[0], inp[1], inp[2], out

    work_dtypes = {"A": dtypeA, "B": dtypeB, "C": dtypeC, "D": dtypeD}

    op = (f"""
dt_$dtypeD$ operation(dt_$dtypeA$ x_inp, dt_$dtypeB$ y_inp, dt_$dtypeC$ z_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    dt_$workB$ y = dt_convert_$dtypeB$_to_$workB$(y_inp);
    dt_$workC$ z = dt_convert_$dtypeC$_to_$workC$(z_inp);
    return dt_{op_name}_$workD$(x, y, z);
}}
""")
    if custom_operation is not None:
        op = custom_operation

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
        dtypeD=dtypeD,
        workA=work_dtypes[workA],
        workB=work_dtypes[workB],
        workC=work_dtypes[workC],
        workD=work_dtypes[workD],
    )
    build_info.add_header(op)
    return build_program(device, "ternary.cl", build_info).binary


def make_ternary(
    device: Device, node: core.SymbolicTensor,
    cast: str = "DDD->D",
    custom_operation: str | None = None
) -> kernel_func_type:
    assert isinstance(node, ternary.Ternary)
    kernel = compile_ternary(
        device,
        (node.x1.dtype, node.x2.dtype, node.x3.dtype),
        node.dtype, node._name, cast,
        custom_operation
    )

    def inner(
        inputs: list[core.ConcreteTensor],
        output: core.ConcreteTensor
    ) -> DeviceEvent:
        x1, x2, x3 = inputs
        return kernel(
            node.size,
            None,
            x1.buffer, x2.buffer, x3.buffer,
            output.buffer,

            get_shape_info(device, x1),
            get_shape_info(device, x2),
            get_shape_info(device, x3),
            get_shape_info(device, output)
        )
    return inner


where_op = """
dt_$dtypeD$ operation(dt_$dtypeA$ x_inp, dt_$dtypeB$ y_inp, dt_$dtypeC$ z_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_bool(x_inp);
    if (x)
        return dt_convert_$dtypeB$_to_$workB$(y_inp);
    else
        return dt_convert_$dtypeC$_to_$workC$(z_inp);
}}
"""
register_impl(ternary.Where)(partial(make_ternary, custom_operation=where_op))
