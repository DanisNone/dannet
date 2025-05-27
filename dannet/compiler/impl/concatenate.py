import dannet as dt
import pyopencl as cl

from .utils import (
    Generator,
    default_strides,
    register_impl,
    build_kernel,
)


def concatenate_part(
    device,
    input, output,
    input_buffer, output_buffer,
    concatenate_offset
):
    gen = Generator()
    gen.nodes_info(A=input, B=output)
    gen.defines(concatenate_offset=concatenate_offset)
    gen.static_array('stridesAN', default_strides(input))

    global_size = (input.size,)
    local_size = None

    kernel = build_kernel(device, 'concatenate.cl', gen)
    return lambda: kernel.concatenate(
        device.queue, global_size, local_size,
        input_buffer, output_buffer
    )


@register_impl(dt.ops.basic._Concatenate)
def concatenate(device, node, input_buffers, output_buffer):
    kernels = []
    offset = 0
    for input, buffer in zip(node.inputs(), input_buffers):
        kernels.append(concatenate_part(
            device, input, node,
            buffer, output_buffer,
            offset
        ))
        offset += input.size

    def run():
        events = []
        for kernel in kernels:
            events.append(kernel())
        return cl.enqueue_marker(device.queue, events)
    return run
