import dannet as dt
from .utils import register_impl


@register_impl(dt.basic._BroadcastTo)
def broadcast_to(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None


@register_impl(dt.basic._Transpose)
def transpose(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None


@register_impl(dt.basic._Flip)
def flip(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None


@register_impl(dt.basic._Slice)
def slice(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None


@register_impl(dt.basic._Bitcast)
def bitcast(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None


@register_impl(dt.basic._Diagonal)
def diagonal(device, node, input_buffers, output_buffer):
    (A,) = input_buffers

    assert A is output_buffer
    return None
