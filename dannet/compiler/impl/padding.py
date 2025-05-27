import dannet as dt

from .utils import (
    Generator,
    build_kernel,
    register_impl
)


@register_impl(dt.basic._Pad)
def pad(
    device,
    node: dt.basic._Pad,
    input_buffers,
    output_buffer,
):
    A, = input_buffers
    B = output_buffer

    assert node._is_contiguous

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)

    pad_left, pad_right = zip(*node._paddings)
    gen.static_array('pad_left', pad_left)
    gen.static_array('pad_right', pad_right)
    gen.mode('zero')

    global_size = (node.size, )
    local_size = None
    kernel = build_kernel(device, 'padding.cl', gen)
    return lambda: kernel.padding(device.queue, global_size, local_size, A, B)
