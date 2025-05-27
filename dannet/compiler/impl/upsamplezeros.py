import dannet as dt

from .utils import (
    Generator,
    register_impl,
    build_kernel
)


@register_impl(dt.nnet.convolve._UpSampleZeros)
def matmul(
    device,
    node: dt.nnet.convolve._UpSampleZeros,
    input_buffers,
    output_buffer,
):
    A,  = input_buffers
    B = output_buffer

    gen = Generator()
    gen.nodes_info(
        A=node.x,
        B=node
    )
    gen.static_array('upsample_size', node._upsample_size)

    global_size = (node.size, )
    local_size = None

    kernel = build_kernel(device, 'upsamplezeros.cl', gen)
    return lambda: kernel.upsamplezeros(
        device.queue,
        global_size,
        local_size,
        A, B
    )
