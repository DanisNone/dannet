from functools import partial
import dannet as dt
from dannet.device import Device, DeviceEvent
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import unary
from dannet.lib.dtypes import DannetDtype
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_unary(
    device: Device,
    inputs: DannetDtype,
    output: DannetDtype,
    op_name: str,
    cast: str = "B->B",
) -> dt.device.DeviceKernel:
    dtypeA = inputs
    dtypeB = output

    workA, workB = cast.split("->")
    assert len(workA) == 1
    assert len(workB) == 1

    dtypes = {"A": dtypeA, "B": dtypeB}

    op = (f"""
dt_$dtypeB$ operation(dt_$dtypeA$ x_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    return dt_{op_name}_$workB$(x);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        workA=dtypes[workA],
        workB=dtypes[workB],
    )
    build_info.add_header(op)
    return build_program(device, "unary.cl", build_info).unary


def make_unary(
    device: Device, node: core.SymbolicTensor,
    cast: str = "B->B"
) -> kernel_func_type:
    assert isinstance(node, unary.Unary)
    kernel = compile_unary(device, node.x.dtype, node.dtype, node._name, cast)

    def inner(
        inputs: list[core.ConcreteTensor],
        output: core.ConcreteTensor
    ) -> DeviceEvent:
        x, = inputs
        return kernel(
            node.size,
            None,
            x.buffer,
            output.buffer,

            get_shape_info(device, x),
            get_shape_info(device, output)
        )
    return inner


register_impl(unary.Negative)(make_unary)
register_impl(unary.Positive)(make_unary)
register_impl(unary.Abs)(partial(make_unary, cast="A->A"))
register_impl(unary.Square)(make_unary)
register_impl(unary.Sqrt)(make_unary)
register_impl(unary.Sign)(make_unary)
register_impl(unary.Conjuagte)(make_unary)

register_impl(unary.Sin)(make_unary)
register_impl(unary.Cos)(make_unary)
register_impl(unary.Tan)(make_unary)
register_impl(unary.Sinh)(make_unary)
register_impl(unary.Cosh)(make_unary)
register_impl(unary.Tanh)(make_unary)

register_impl(unary.Arcsin)(make_unary)
register_impl(unary.Arccos)(make_unary)
register_impl(unary.Arctan)(make_unary)
register_impl(unary.Arcsinh)(make_unary)
register_impl(unary.Arccosh)(make_unary)
register_impl(unary.Arctanh)(make_unary)

register_impl(unary.Exp)(make_unary)
register_impl(unary.Exp2)(make_unary)
register_impl(unary.Exp10)(make_unary)
register_impl(unary.Expm1)(make_unary)

register_impl(unary.Log)(make_unary)
register_impl(unary.Log2)(make_unary)
register_impl(unary.Log10)(make_unary)
register_impl(unary.Log1p)(make_unary)
