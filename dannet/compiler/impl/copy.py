import dannet as dt
from .utils import (
    Generator,
    default_strides,
    build_kernel,
    register_impl
)


@register_impl(dt.basic._Copy)
def copy(device, node: dt.basic._Copy, input_buffers, output_buffer):
    assert node._is_contiguous

    (A,) = input_buffers
    B = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)
    gen.static_array('stridesAN', default_strides(node.x))
    gen.mode('strided')

    kernel = build_kernel(device, 'copy.cl', gen)
    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(
        device.queue, global_size, local_size,
        A, B
    )


@register_impl(dt.basic._Cast)
def cast(device, node: dt.basic._Cast, input_buffers, output_buffer):
    assert node._is_contiguous

    (A,) = input_buffers
    B = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)
    gen.static_array('stridesAN', default_strides(node.x))
    gen.mode('strided')

    kernel = build_kernel(device, 'copy.cl', gen)
    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(
        device.queue, global_size, local_size,
        A, B
    )


@register_impl(dt.basic._Reshape)
def reshape(device, node: dt.basic._Reshape, input_buffers, output_buffer):
    assert node._is_contiguous

    (A,) = input_buffers
    B = output_buffer

    if A is B:
        assert node._buffer_offset == node.x._buffer_offset
        return None

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)
    gen.static_array('stridesAN', default_strides(node.x))
    gen.mode('strided')

    kernel = build_kernel(device, 'copy.cl', gen)
    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(
        device.queue, global_size, local_size,
        A, B
    )


@register_impl(dt.core.Update)
def update(device, node: dt.core.Update, input_buffers, output_buffer):
    assert node._variable._is_contiguous
    var, value = input_buffers
    assert var is output_buffer

    gen = Generator()
    gen.nodes_info(A=node._value, B=node._variable)
    gen.static_array('stridesAN', default_strides(node._value))
    gen.mode('strided')

    kernel = build_kernel(device, 'copy.cl', gen)
    global_size = (node.size,)
    local_size = None
    return lambda: kernel.copy_strided(
        device.queue, global_size, local_size,
        value, var
    )
