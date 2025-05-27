import dannet as dt
from .utils import (
    Generator,
    default_strides,
    build_kernel,
    register_impl
)


@register_impl(dt.basic._OneHot)
def onehot(device, node: dt.basic._OneHot, input_buffers, output_buffer):
    A, = input_buffers
    B = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.indices, B=node)
    gen.static_array('stridesAN', default_strides(node.indices.shape))

    global_size = (node.indices.size,)
    local_size = None
    kernel = build_kernel(device, 'onehot.cl', gen)
    return lambda: kernel.onehot(device.queue, global_size, local_size, A, B)
