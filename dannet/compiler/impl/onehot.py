import dannet as dt
from .utils import *


@register_impl(dt.basic._OneHot)
def gather_impl(device, node: dt.basic._OneHot, input_buffers, output_buffer):
    A, = input_buffers
    B = output_buffer

    headers = generate_nodes_info(A=node.indices, B=node)
    headers.append(insert_static_array("stridesAN", default_strides(node.indices.shape)))
    
    global_size = (node.indices.size,)
    local_size = None
    kernel = build_kernel(device, 'onehot.cl', headers)
    return lambda: kernel.onehot(device.queue, global_size, local_size, A, B)
