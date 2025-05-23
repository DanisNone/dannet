import math
import dannet as dt
from .utils import (
    generate_nodes_info,
    generate_defines,
    generate_static_array,
    generate_mode,
    default_strides,
    build_kernel,
    register_impl
)


def arg_reduce_full(
    device,
    node: dt.reduce._ArgReduce,
    input_buffers,
    output_buffer,
    condition: str,
):
    A, = input_buffers
    B = output_buffer

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(
        generate_static_array('stridesAN', default_strides(node.x))
    )
    headers.append(generate_mode('full'))
    headers.append(f'''
bool condition(dtypeA x, dtypeA y)
{{
    return {condition};
}}
''')

    global_size = (1, )
    local_size = None

    kernel = build_kernel(device, 'arg_reduce.cl', headers)
    return lambda: kernel.reduce(device.queue, global_size, local_size, A, B)


def arg_reduce(
    device,
    node: dt.reduce._ArgReduce,
    input_buffers,
    output_buffer,
    condition: str,
):
    assert node._is_contiguous
    (A,) = input_buffers
    B = output_buffer

    assert node.x.size % node.size == 0

    if node._axis is None:
        return arg_reduce_full(
            device, node, input_buffers, output_buffer, condition
        )

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(
        generate_static_array('stridesAN', default_strides(node.x))
    )
    headers.extend(generate_defines(
        skeep_axis=node._axis,
        sizeRight=math.prod(node.x._shape[node._axis+1:]))
    )
    headers.append(generate_mode('by_axis'))
    headers.append(f'''
bool condition(dtypeA x, dtypeA y)
{{
    return {condition};
}}
''')

    global_size = (node.size, )
    local_size = None

    kernel = build_kernel(device, 'arg_reduce.cl', headers)
    return lambda: kernel.reduce(device.queue, global_size, local_size, A, B)


@register_impl(dt.reduce._ArgMin)
def min(device, node, input_buffers, output_buffer):
    return arg_reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        'x > y',
    )


@register_impl(dt.reduce._ArgMax)
def max(device, node, input_buffers, output_buffer):
    return arg_reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        'x < y',
    )
