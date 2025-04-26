from .utils import *


@register_impl(dt.basic._Zeros)
def zeros(device, node: dt.basic._Zeros, input_buffers, output_buffer):
    assert node._is_default_strides()

    A = output_buffer

    headers = generate_nodes_info(A=node)
    headers.append(generate_mode('zeros'))
    kernel = build_kernel(device, 'const_fill.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.fill(device.queue, global_size, local_size, A)


@register_impl(dt.basic._Ones)
def ones(device, node: dt.basic._Ones, input_buffers, output_buffer):
    assert node._is_default_strides()

    A = output_buffer

    headers = generate_nodes_info(A=node)
    headers.append(generate_mode('ones'))
    kernel = build_kernel(device, 'const_fill.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.fill(device.queue, global_size, local_size, A)
