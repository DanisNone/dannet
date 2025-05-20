import dannet as dt
import pyopencl as cl

from .utils import (
    default_strides,
    generate_nodes_info,
    generate_defines,
    generate_static_array,
    register_impl,
    build_kernel,
)


def concatenate_part(
    device,
    input, output,
    input_buffer, output_buffer,
    concatenate_offset
):
    headers = generate_nodes_info(A=input, B=output)
    headers.extend(generate_defines(concatenate_offset=concatenate_offset))
    headers.append(generate_static_array('stridesAN', default_strides(input)))
    global_size = (input.size, )
    local_size = None

    kernel = build_kernel(device, 'concatenate.cl', headers)
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
