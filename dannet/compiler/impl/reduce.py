import dannet as dt
from .utils import (
    Generator,
    default_strides,
    register_impl,
    build_kernel
)


def reduce(
    device,
    node: dt.reduce._Reduce,
    input_buffers,
    output_buffer,
    *,
    op: str,
    init_op: str | None = None,
    final_op: str | None = None
):
    assert node._is_contiguous
    (A,) = input_buffers
    B = output_buffer

    assert node.x.size % node.size == 0

    inner_size = node.x.size // node.size
    inner_shape = [s for i, s in enumerate(node.x._shape) if i in node._axis]
    inner_strides = [
        s
        for i, s in enumerate(node.x._strides)
        if i in node._axis
    ]
    inner_strides_norm = default_strides(inner_shape)

    outer_size = node.size
    outer_shape = [
        s
        for i, s in enumerate(node.x._shape)
        if i not in node._axis
    ]
    outer_strides = [
        s
        for i, s in enumerate(node.x._strides)
        if i not in node._axis
    ]
    outer_strides_norm = default_strides(outer_shape)

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)

    gen.static_array('shapeI', inner_shape)
    gen.static_array('stridesI', inner_strides)
    gen.static_array('stridesIN', inner_strides_norm)
    gen.defines(ndimI=len(inner_shape), sizeI=inner_size)

    gen.static_array('shapeO', outer_shape)
    gen.static_array('stridesO', outer_strides)
    gen.static_array('stridesON', outer_strides_norm)
    gen.defines(ndimO=len(outer_shape), sizeO=outer_size)

    if init_op is None:
        init_op = 'x'
    if final_op is None:
        final_op = 'res'

    gen.line(f'''
dtypeB init_operation(dtypeA x_inp)
{{
    dtypeB x = dt_convert_dtypeA_to_dtypeB(x_inp);
    return {init_op};
}}

dtypeB operation(dtypeB acc, dtypeA x_inp)
{{
    dtypeB x = dt_convert_dtypeA_to_dtypeB(x_inp);
    return {op};
}}

dtypeB final_operation(dtypeB res, size_t inner_size)
{{
    return {final_op};
}}
''')

    gen.mode('general')

    global_size = (node.size,)
    local_size = None
    kernel = build_kernel(device, 'reduce.cl', gen)
    return lambda: kernel.reduce(device.queue, global_size, local_size, A, B)


@register_impl(dt.reduce._Sum)
def sum(device, node, input_buffers, output_buffer):
    return reduce(
        device, node, input_buffers, output_buffer,
        op='dt_add_dtypeB(acc, x)',
    )


@register_impl(dt.reduce._DefaultDtypeSum)
def default_typed_sum(device, node, input_buffers, output_buffer):
    return reduce(
        device, node, input_buffers, output_buffer,
        op='dt_add_dtypeB(acc, x)',
    )


@register_impl(dt.reduce._Mean)
def mean(device, node, input_buffers, output_buffer):
    final_op = 'dt_divide_dtypeB(res, dt_convert_uint64_to_dtypeB(inner_size))'
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='dt_add_dtypeB(acc, x)',
        final_op=final_op
    )


@register_impl(dt.reduce._Prod)
def prod(device, node, input_buffers, output_buffer):
    return reduce(
        device, node, input_buffers, output_buffer,
        op='dt_multiply_dtypeB(acc, x)',
    )


@register_impl(dt.reduce._Min)
def min(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='dt_min_dtypeB(acc, x)',
    )


@register_impl(dt.reduce._Max)
def max(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='dt_max_dtypeB(acc, x)',
    )


@register_impl(dt.reduce._Any)
def any(device, node, input_buffers, output_buffer):
    assert node.dtype == dt.dtype.bool_dtype
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='acc || x',
    )


@register_impl(dt.reduce._All)
def all(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='acc && x',
    )


@register_impl(dt.nnet.activations._LogSumExp)
def logsumexp(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        op='dt_add_dtypeB(acc, dt_exp_dtypeB(x))',
        init_op='dt_exp(x)',
        final_op='dt_log(res)'
    )
