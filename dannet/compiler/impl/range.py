from .utils import *


@register_impl(dt.basic._Range)
def binary(
    device, node: dt.basic._Range, input_buffers, output_buffer
):
    assert node._is_default_strides()

    A = output_buffer

    headers = generate_nodes_info(A=node)
        
    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'range.cl', headers)
    return lambda: kernel.range(device.queue, global_size, local_size, A)