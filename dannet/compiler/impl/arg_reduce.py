import math
import dannet as dt
from .utils import (
    Generator,
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

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)
    gen.static_array('stridesAN', default_strides(node.x))
    gen.mode('full')
    gen.line(f'''
bool condition(dtypeA x, dtypeA y)
{{
    return {condition}(x, y);
}}
''')

    global_size = (1, )
    local_size = None

    kernel = build_kernel(device, 'arg_reduce.cl', gen)
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

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)
    gen.static_array('stridesAN', default_strides(node.x))
    gen.defines(
        skeep_axis=node._axis,
        sizeRight=math.prod(node.x._shape[node._axis + 1:])
    )
    gen.mode('by_axis')
    gen.line(f'''
dt_bool condition(dtypeA x, dtypeA y)
{{
    return {condition}(x, y);
}}''')

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'arg_reduce.cl', gen)
    return lambda: kernel.reduce(device.queue, global_size, local_size, A, B)


@register_impl(dt.reduce._ArgMin)
def min(device, node, input_buffers, output_buffer):
    return arg_reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        'dt_greater_dtypeA',
    )


@register_impl(dt.reduce._ArgMax)
def max(device, node, input_buffers, output_buffer):
    return arg_reduce(
        device,
        node,
        input_buffers,
        output_buffer,
        'dt_less_dtypeA',
    )
