import dannet as dt
from .utils import (
    Generator,
    default_strides,
    build_kernel,
    register_impl
)


@register_impl(dt.basic._Gather)
def gather_impl(device, node: dt.basic._Gather, input_buffers, output_buffer):
    A, B = input_buffers
    C = output_buffer

    stridesBN = default_strides(node.indices.shape)
    stridesON = default_strides(node.shape)

    gen = Generator()

    gen.nodes_info(A=node.x, B=node.indices, C=node)

    gen.static_array('stridesBN', stridesBN)
    gen.static_array('stridesON', stridesON)

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'gather.cl', gen)
    return lambda: kernel.gather(
        device.queue, global_size, local_size,
        A, B, C
    )
