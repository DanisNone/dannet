import dannet as dt

from .utils import (
    generate_nodes_info,
    generate_mode,
    build_kernel,
    register_impl
)


def ternary(
    device,
    node: dt.math._ElementWiseTernary,
    input_buffers,
    output_buffer,
    op: str
):
    assert node._is_contiguous

    A, B, C = input_buffers
    D = output_buffer

    headers = generate_nodes_info(A=node.x, B=node.y, C=node.z, D=node)
    headers.append(
        f'''
dtypeD operation(dtypeA x, dtypeB y, dtypeC z)
{{
    return {op};
}}
'''
    )
    headers.append(generate_mode('strided'))
    kernel = build_kernel(device, 'ternary.cl', headers)

    global_size = (node.size,)
    local_size = None
    return lambda: kernel.general(
        device.queue, global_size, local_size,
        A, B, C, D
    )


@register_impl(dt.math._Where)
def where(device, node, input_buffers, output_buffer):
    return ternary(
        device, node, input_buffers, output_buffer,
        '(bool)x ? (dtypeD)y : (dtypeD)z'
    )


@register_impl(dt.math._Clip)
def clip(device, node, input_buffers, output_buffer):
    return ternary(
        device, node, input_buffers, output_buffer,
        '((x > y ? x : y) < z ? (x > y ? x : y) : z)'
    )
