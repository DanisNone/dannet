import dannet as dt

from .utils import (
    generate_nodes_info,
    generate_mode,
    default_strides,
    generate_static_array,
    generate_defines,
    register_impl,
    build_kernel
)


def reduce(
    device,
    node: dt.reduce._Reduce,
    input_buffers,
    output_buffer,
    init_value: str,
    operation: str,
    final_operation: str,
):
    assert node._is_default_strides()
    (A,) = input_buffers
    B = output_buffer

    assert node.x.size % node.size == 0

    inner_size = node.x.size // node.size
    inner_shape = [s for i, s in enumerate(node.x._shape) if i in node._axis]
    inner_strides = [s for i, s in enumerate(
        node.x._strides) if i in node._axis]
    inner_strides_norm = default_strides(inner_shape)

    outer_size = node.size
    outer_shape = [s for i, s in enumerate(
        node.x._shape) if i not in node._axis]
    outer_strides = [s for i, s in enumerate(
        node.x._strides) if i not in node._axis]
    outer_strides_norm = default_strides(outer_shape)

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(generate_static_array('shapeI', inner_shape))
    headers.append(generate_static_array('stridesI', inner_strides))
    headers.append(generate_static_array('stridesIN', inner_strides_norm))
    headers.extend(generate_defines(ndimI=len(inner_shape), sizeI=inner_size))

    headers.append(generate_static_array('shapeO', outer_shape))
    headers.append(generate_static_array('stridesO', outer_strides))
    headers.append(generate_static_array('stridesON', outer_strides_norm))
    headers.extend(generate_defines(ndimO=len(outer_shape), sizeO=outer_size))

    headers.append(
        f'''
__constant dtypeB init_value = {init_value};

dtypeB operation(dtypeB acc, dtypeA x)
{{
    return {operation};
}}

dtypeB final_operation(dtypeB res, size_t inner_size)
{{
    return {final_operation};
}}
'''
    )
    headers.append(generate_mode('general'))

    global_size = (node.size,)
    local_size = None
    kernel = build_kernel(device, 'reduce.cl', headers)
    return lambda: kernel.reduce(device.queue, global_size, local_size, A, B)


@register_impl(dt.reduce._Sum)
def sum(device, node, input_buffers, output_buffer):
    return reduce(
        device, node, input_buffers, output_buffer,
        '0', 'acc + x', 'res'
    )


@register_impl(dt.reduce._Mean)
def mean(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        '0',
        'acc + x',
        'res / (dtypeB)inner_size',
    )


@register_impl(dt.reduce._Prod)
def prod(device, node, input_buffers, output_buffer):
    return reduce(
        device, node, input_buffers, output_buffer,
        '1', 'acc * (dtypeB)x', 'res'
    )


@register_impl(dt.reduce._Min)
def min(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        'INFINITY',
        'acc < x ? acc : x',
        'res',
    )


@register_impl(dt.reduce._Max)
def max(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        '-INFINITY',
        'acc > x ? acc : x',
        'res',
    )


@register_impl(dt.nnet.activations._LogSumExp)
def logsumexp(device, node, input_buffers, output_buffer):
    return reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        '0',
        'acc + exp((dtypeB)x)',
        'log(res)',
    )
