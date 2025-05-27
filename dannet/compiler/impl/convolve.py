import dannet as dt
from .utils import (
    Generator,
    build_kernel,
    register_impl
)


@register_impl(dt.nnet.convolve._ConvND)
def conv(device, node: dt.nnet.convolve._ConvND, input_buffers, output_buffer):
    assert node._is_contiguous

    A, B = input_buffers
    C = output_buffer
    assert 1 <= node.rank <= 3
    gen = Generator()
    gen.nodes_info(A=node.input, B=node.kernel, C=node)
    gen.static_array('stride', node._conv_strides)

    gen.mode(f'conv{node.rank}d')
    kernel = build_kernel(device, 'convolve.cl', gen)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.conv(device.queue, global_size, local_size, A, B, C)


@register_impl(dt.nnet.convolve._DepthwiseConv2D)
def depthwise_conv(
    device,
    node: dt.nnet.convolve._DepthwiseConv2D,
    input_buffers,
    output_buffer
):
    assert node._is_contiguous

    A, B = input_buffers
    C = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.input, B=node.kernel, C=node)
    gen.static_array('stride', node._conv_strides)

    gen.mode('depthwise_conv2d')
    kernel = build_kernel(device, 'convolve.cl', gen)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.depthwise_conv(
        device.queue, global_size, local_size,
        A, B, C
    )
