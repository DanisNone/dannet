from .utils import *


@register_impl(dt.basic._Copy)
def copy(device, node: dt.basic._Copy, input_buffers, output_buffer):
    assert node._is_default_strides()

    (A,) = input_buffers
    B = output_buffer

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(insert_static_array("stridesAN", node.x._default_strides()))
    headers.append(generate_mode('strided'))
    kernel = build_kernel(device, 'copy.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(device.queue, global_size, local_size, A, B)

@register_impl(dt.basic._Cast)
def cast(device, node: dt.basic._Cast, input_buffers, output_buffer):
    assert node._is_default_strides()

    (A,) = input_buffers
    B = output_buffer

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(insert_static_array("stridesAN", node.x._default_strides()))
    headers.append(generate_mode('strided'))
    kernel = build_kernel(device, 'copy.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(device.queue, global_size, local_size, A, B)

# TODO: implement smart reshape
@register_impl(dt.basic._Reshape)
def reshape(device, node: dt.basic._Reshape, input_buffers, output_buffer):
    assert node._is_default_strides()
    (A,) = input_buffers
    B = output_buffer


    if A is B:
        assert node._buffer_offset == node.x._buffer_offset
        return lambda:None

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(insert_static_array("stridesAN", node.x._default_strides()))
    headers.append(generate_mode('strided'))
    kernel = build_kernel(device, 'copy.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(device.queue, global_size, local_size, A, B)


@register_impl(dt.core.Update)
def update(device, node: dt.core.Update, input_buffers, output_buffer):
    assert node._variable._is_default_strides()
    var, value = input_buffers
    assert var is output_buffer

    headers = generate_nodes_info(A=node._value, B=node._variable)
    headers.append(insert_static_array("stridesAN", node._value._default_strides()))
    headers.append(generate_mode('strided'))
    kernel = build_kernel(device, 'copy.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(
        device.queue, global_size, local_size, value, var
    )

@register_impl(dt.basic._BroadcastTo)
def broadcast_to(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return lambda: None

@register_impl(dt.basic._Transpose)
def transpose(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return lambda: None

@register_impl(dt.basic._Flip)
def flip(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return lambda: None
    
@register_impl(dt.basic._Slice)
def slice(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return lambda: None