import dannet as dt
from dannet.device import Device, DeviceEvent, DeviceKernel
from dannet.compiler.register import register_impl, kernel_func_type
from dannet.lib import core
from dannet.lib import reductions
from dannet.lib.dtypes import DannetDtype
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape,
    get_shape_info,
    get_size_t
)


def compiler_reduction(
    device: Device,
    inputs: DannetDtype,
    output: DannetDtype,
    main_op: str,
    *,
    init_op: str,
    final_op: str,
    mean_normalize: bool = False,
    cast: str = "B->B",
) -> dt.device.DeviceKernel:
    dtypeA = inputs
    dtypeB = output

    workA, workB = cast.split("->")
    assert len(workA) == 1
    assert len(workB) == 1

    dtypes = {"A": dtypeA, "B": dtypeB}

    op = (f"""
dt_$dtypeB$ init(dt_$dtypeA$ x_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    return dt_{init_op}_$workB$(x);
}}

dt_$dtypeB$ operation(dt_$dtypeB$ acc, dt_$dtypeA$ x_inp)
{{
    dt_$workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    return dt_{main_op}_$workB$(acc, x);
}}

""")
    if mean_normalize:
        final = f"""
dt_$dtypeB$ final(dt_$dtypeB$ acc, dt_$size_t$ size)
{{
    dt_$dtypeB$ tmp = dt_{final_op}_$workB$(acc);
    dt_$dtypeB$ size_ = dt_convert_$size_t$_to_$dtypeB$(size);
    return dt_divide_$dtypeB$(tmp, size_);
}}
"""
    else:
        final = f"""
dt_$dtypeB$ final(dt_$dtypeB$ acc, dt_$size_t$ size)
{{
    return dt_{final_op}_$workB$(acc);
}}
"""

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        workA=dtypes[workA],
        workB=dtypes[workB],
    )
    build_info.add_header(op)
    build_info.add_header(final)
    return build_program(device, "reduction.cl", build_info).reduction


def make_reductions(
    device: Device, kernel: DeviceKernel,
    node: core.SymbolicTensor
) -> kernel_func_type:
    assert isinstance(node, reductions.Reduction)
    inner_strides = core.default_strides(node.x.shape[node.ndim:])

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
            get_shape_info(device, output),
            get_shape(device, inner_strides),
            get_size_t(device, node._inner_size),
        )
    return inner


@register_impl(reductions.Sum)
def sum(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, reductions.Sum)
    kernel = compiler_reduction(
        device, node.x.dtype,
        node.dtype,
        "add",
        init_op="positive",
        final_op="positive"
    )

    return make_reductions(device, kernel, node)


@register_impl(reductions.Mean)
def mean(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, reductions.Mean)
    kernel = compiler_reduction(
        device, node.x.dtype,
        node.dtype,
        "add",
        init_op="positive",
        final_op="positive",
        mean_normalize=True
    )

    return make_reductions(device, kernel, node)


@register_impl(reductions.Prod)
def prod(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, reductions.Prod)
    kernel = compiler_reduction(
        device, node.x.dtype,
        node.dtype,
        "multiply",
        init_op="positive",
        final_op="positive",
    )

    return make_reductions(device, kernel, node)


@register_impl(reductions.Min)
def min(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, reductions.Min)
    kernel = compiler_reduction(
        device, node.x.dtype,
        node.dtype,
        "min",
        init_op="positive",
        final_op="positive",
    )

    return make_reductions(device, kernel, node)


@register_impl(reductions.Max)
def max(device: Device, node: core.SymbolicTensor) -> kernel_func_type:
    assert isinstance(node, reductions.Max)
    kernel = compiler_reduction(
        device, node.x.dtype,
        node.dtype,
        "max",
        init_op="positive",
        final_op="positive",
    )

    return make_reductions(device, kernel, node)
