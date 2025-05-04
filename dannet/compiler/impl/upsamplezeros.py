import dannet as dt

from .utils import *


@register_impl(dt.nnet.convolve._UpSampleZeros)
def matmul(
    device,
    node: dt.nnet.convolve._UpSampleZeros,
    input_buffers,
    output_buffer,
):
    A,  = input_buffers
    B = output_buffer

    headers = generate_nodes_info(
        A=node.x,
        B=node
    )
    headers.append(insert_static_array('upsample_size', node._upsample_size))
    
    global_size = (node.size, )
    local_size = None

    kernel = build_kernel(device, 'upsamplezeros.cl', headers)
    return lambda: kernel.upsamplezeros(device.queue, global_size, local_size, A, B)
