import dannet as dt
from .utils import (
    generate_nodes_info,
    generate_mode,
    build_kernel,
    register_impl
)


def binary(
    device,
    node: dt.math._ElementWiseBinary,
    input_buffers,
    output_buffer,
    op: str,
    headers: list[str] | None = None
):
    assert node._is_contiguous

    A, B = input_buffers
    C = output_buffer

    headers = [] if headers is None else headers
    headers = generate_nodes_info(A=node.x, B=node.y, C=node) + headers
    headers.append(
        f'''
dtypeC operation(dtypeA x, dtypeB y)
{{
    return {op};
}}
'''
    )

    if node.x._is_contiguous and node.y._is_contiguous:
        headers.append(generate_mode('full'))
    else:
        headers.append(generate_mode('strided'))

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'binary.cl', headers)
    return lambda: kernel.binary(
        device.queue, global_size, local_size,
        A, B, C
    )


@register_impl(dt.math._Add)
def add(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x + (dtypeC)y'
    )


@register_impl(dt.math._Subtract)
def subtract(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x - (dtypeC)y'
    )


@register_impl(dt.math._Multiply)
def multiply(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x * (dtypeC)y'
    )


@register_impl(dt.math._Divide)
def divide(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x / (dtypeC)y'
    )


@register_impl(dt.math._Power)
def power(device, node, input_buffers, output_buffer):
    if not dt.dtype.is_float_dtype(node.dtype):
        headers = ['''
dtypeC int_pow(dtypeC base, dtypeC exp) {
    dtypeC result = 1;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
}
''']
        return binary(
            device, node, input_buffers, output_buffer,
            'int_pow(x, y)', headers
        )
    return binary(
        device, node, input_buffers, output_buffer,
        'pow((dtypeC)x, (dtypeC)y)'
    )


@register_impl(dt.math._Minimum)
def minimum(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x < y ? x : y')


@register_impl(dt.math._Maximum)
def maximum(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x < y ? y : x')


@register_impl(dt.math._FloorDivide)
def floor_divide(device, node, input_buffers, output_buffer):
    if dt.dtype.is_float_dtype(node.dtype):
        header = """
dtypeC floor_divide(dtypeA x, dtypeB y)
{
    return floor(x / y);
}
"""
    else:
        header = """
dtypeC floor_divide(dtypeA x, dtypeB y)
{
    dtypeC q = (dtypeC)x / (dtypeC)y;
    dtypeC r = (dtypeC)x % (dtypeC)y;
    if ((r != 0) && ((x < 0) != (y < 0))) {
        q -= 1;
    }
    return q;
}
"""
    return binary(
        device, node, input_buffers, output_buffer,
        'floor_divide(x, y)', [header]
    )


@register_impl(dt.logical._Equal)
def equal(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x == y')


@register_impl(dt.logical._NotEqual)
def not_equal(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x != y')


@register_impl(dt.logical._Greater)
def greater(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x > y')


@register_impl(dt.logical._GreaterEqual)
def greater_equal(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x >= y')


@register_impl(dt.logical._Less)
def less(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x < y')


@register_impl(dt.logical._LessEqual)
def less_equal(device, node, input_buffers, output_buffer):
    return binary(device, node, input_buffers, output_buffer, 'x <= y')


@register_impl(dt.bitwise._BitwiseOr)
def bitwise_or(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x | (dtypeC)y'
    )


@register_impl(dt.bitwise._BitwiseAnd)
def bitwise_and(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x & (dtypeC)y'
    )


@register_impl(dt.bitwise._BitwiseXor)
def bitwise_xor(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(dtypeC)x ^ (dtypeC)y'
    )


@register_impl(dt.logical._LogicalOr)
def logical_or(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(bool)x || (bool)y'
    )


@register_impl(dt.logical._LogicalAnd)
def logical_and(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(bool)x && (bool)y'
    )


@register_impl(dt.logical._LogicalXor)
def logical_xor(device, node, input_buffers, output_buffer):
    return binary(
        device, node, input_buffers, output_buffer,
        '(bool)x != (bool)y'
    )
