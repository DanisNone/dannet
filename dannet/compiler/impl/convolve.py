from .utils import *


def conv(device, node: dt.nnet.convolve._ConvND, input_buffers, output_buffer, rank, depthwise=False):
    assert node.x._is_default_strides()
    assert node.kernel._is_default_strides()
    assert node._is_default_strides()

    A, B = input_buffers
    C = output_buffer
    assert 1 <= rank <= 3
    headers = generate_nodes_info(A=node.x, B=node.kernel, C=node)
    headers.append(insert_static_array("stride", node.stride))

    if depthwise:
        headers.append(generate_mode(f'conv{rank}d_depthwise'))
    else:
        headers.append(generate_mode(f'conv{rank}d'))
    
    kernel = build_kernel(device, 'convolve.cl', headers)
    kernel.conv
    global_size = (node.size,)
    local_size = None
    return lambda: kernel.conv(device.queue, global_size, local_size, A, B, C)

@register_impl(dt.nnet.convolve._Conv1D)
def conv1d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 1)

@register_impl(dt.nnet.convolve._Conv2D)
def conv2d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 2)

@register_impl(dt.nnet.convolve._Conv3D)
def conv3d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 3)


@register_impl(dt.nnet.convolve._DepthwiseConv1D)
def depthwise_conv1d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 1, True)

@register_impl(dt.nnet.convolve._DepthwiseConv2D)
def depthwise_conv2d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 2, True)

@register_impl(dt.nnet.convolve._DepthwiseConv3D)
def depthwise_conv3d(device, node, input_buffers, output_buffer):
    return conv(device, node, input_buffers, output_buffer, 3, True)
