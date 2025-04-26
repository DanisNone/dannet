import dannet as dt

from .utils import *


@register_impl(dt.random._RandomInt)
def random_int(device, node: dt.random._RandomInt, input_buffers, output_buffer):
    A = output_buffer
    assert node._dtype == 'uint64'

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'random.cl')
    return lambda: kernel.random(
        device.queue, global_size, local_size, A, node.rng.get_seed(node.size)
    )
@register_impl(dt.random._RandomFloat)
def random_float(device, node: dt.random._RandomFloat, input_buffers, output_buffer):
    A = output_buffer
    assert node._dtype in ['float32', 'float64']

    global_size = (node.size,)
    local_size = None

    headers = generate_nodes_info(A=node)
    kernel = build_kernel(device, 'random.cl', headers)
    return lambda: kernel.random_float(
        device.queue, global_size, local_size, A, node.rng.get_seed(node.size)
    )