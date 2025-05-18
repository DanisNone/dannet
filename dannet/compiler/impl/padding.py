import dannet as dt

from .utils import (
    generate_nodes_info,
    generate_static_array,
    generate_mode,
    build_kernel,
    register_impl
)


@register_impl(dt.basic._Pad)
def matmul(
    device,
    node: dt.basic._Pad,
    input_buffers,
    output_buffer,
):
    A, = input_buffers
    B = output_buffer

    assert node._is_contiguous

    headers = generate_nodes_info(
        A=node.x,
        B=node
    )
    pad_left, pad_right = zip(*node._paddings)
    headers.append(generate_static_array('pad_left', pad_left))
    headers.append(generate_static_array('pad_right', pad_right))
    headers.append(generate_mode('zero'))

    global_size = (node.size, )
    local_size = None
    kernel = build_kernel(device, 'padding.cl', headers)
    return lambda: kernel.padding(device.queue, global_size, local_size, A, B)
