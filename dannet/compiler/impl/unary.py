import dannet as dt
from .utils import (
    generate_nodes_info,
    generate_mode,
    register_impl,
    build_kernel,
    to_cl_dtype
)


def unary(
    device,
    node: dt.math._ElementWiseUnary,
    input_buffers,
    output_buffer,
    op: str
):
    (A,) = input_buffers
    B = output_buffer

    headers = generate_nodes_info(A=node.x, B=node)
    headers.append(
        f'''
dtypeB operation(dtypeA x)
{{
    return {op};
}}
'''
    )
    if node.x._is_contiguous:
        headers.append(generate_mode('full'))
    else:
        headers.append(generate_mode('strided'))

    global_size = (node.size,)
    local_size = None
    kernel = build_kernel(device, 'unary.cl', headers)

    return lambda: kernel.general(device.queue, global_size, local_size, A, B)


@register_impl(dt.math._Negative)
def negative(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, '-x')


@register_impl(dt.math._Reciprocal)
def reciprocal(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, '1 / x')


@register_impl(dt.math._Square)
def square(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x * x')


@register_impl(dt.math._Abs)
def abs(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'x >= (dtypeA)0? x : -x'
    )


@register_impl(dt.math._Sign)
def sign(device, node, input_buffers, output_buffer):
    return unary(
        device,
        node,
        input_buffers,
        output_buffer,
        'x > (dtypeA)0 ? 1 : (x < (dtypeA)0 ? -1 : 0)',
    )


@register_impl(dt.math._Exp)
def exp(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'exp((dtypeB)x)'
    )


@register_impl(dt.math._Exp2)
def exp2(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'exp2((dtypeB)x)'
    )


@register_impl(dt.math._Exp10)
def exp10(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'exp10((dtypeB)x)'
    )


@register_impl(dt.math._Expm1)
def expm1(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'expm1((dtypeB)x)'
    )


@register_impl(dt.math._Log)
def log(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'log((dtypeB)x)'
    )


@register_impl(dt.math._Log2)
def log2(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'log2((dtypeB)x)'
    )


@register_impl(dt.math._Log10)
def log10(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'log10((dtypeB)x)'
    )


@register_impl(dt.math._Log1p)
def log1p(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'log1p((dtypeB)x)'
    )


@register_impl(dt.math._Sqrt)
def sqrt(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'sqrt((dtypeB)x)')


@register_impl(dt.math._Rsqrt)
def rsqrt(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        '1.0 / sqrt((dtypeB)x)'
    )


@register_impl(dt.math._Sin)
def sin(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'sin((dtypeB)x)')


@register_impl(dt.math._Cos)
def cos(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'cos((dtypeB)x)')


@register_impl(dt.math._Tan)
def tan(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'tan((dtypeB)x)')


@register_impl(dt.math._Sinh)
def sinh(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'sinh((dtypeB)x)')


@register_impl(dt.math._Cosh)
def cosh(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'cosh((dtypeB)x)')


@register_impl(dt.math._Tanh)
def tanh(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'tanh((dtypeB)x)')


@register_impl(dt.math._Arcsin)
def arcsin(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'asin((dtypeB)x)')


@register_impl(dt.math._Arccos)
def arccos(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'acos((dtypeB)x)')


@register_impl(dt.math._Arctan)
def arctan(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'atan((dtypeB)x)')


@register_impl(dt.math._Arcsinh)
def arcsinh(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'asinh((dtypeB)x)'
    )


@register_impl(dt.math._Arccosh)
def arccosh(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'acosh((dtypeB)x)'
    )


@register_impl(dt.math._Arctanh)
def arctanh(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer, 'atanh((dtypeB)x)'
    )


@register_impl(dt.math._Round)
def round(device, node, input_buffers, output_buffer):
    assert dt.dtype.is_float_dtype(node.dtype)
    bits = node.itemsize * 8
    out = dt.dtype.normalize_dtype(f'int{bits}')
    dtype = to_cl_dtype(out)
    return unary(
        device, node, input_buffers, output_buffer, f'convert_{dtype}_rte(x)'
    )


@register_impl(dt.math._Trunc)
def trunc(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'trunc(x)')


@register_impl(dt.math._Floor)
def floor(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'floor(x)')


@register_impl(dt.math._Ceil)
def ceil(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'ceil(x)')


@register_impl(dt.nnet.activations._Relu)
def relu(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x > 0 ? x : 0')


@register_impl(dt.nnet.activations._Relu6)
def relu6(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'clamp((dtypeB)x, (dtypeB)0, (dtypeB)6)'
    )


@register_impl(dt.nnet.activations._Sigmoid)
def sigmoid(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        '1.0 / (1 + exp(-(dtypeB)x))'
    )


@register_impl(dt.nnet.activations._Softplus)
def softplus(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'log(1 + exp((dtypeB)x))'
    )


@register_impl(dt.nnet.activations._Softsign)
def softsign(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'x / (1.0 + (dtypeB)(x < 0?-x:x))'
    )


@register_impl(dt.nnet.activations._HardSigmoid)
def hard_sigmoid(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'x < -3 ? 0 : (x > 3? 1 : (dtypeB)x * (1.0 / 6.0) + 0.5)'
    )


@register_impl(dt.bitwise._BitwiseNot)
def bitwise_not(device, node, input_buffers, output_buffer):
    op = '~x'
    if dt.dtype.is_bool_dtype(node.x.dtype):
        op = '!x'
    return unary(
        device, node, input_buffers, output_buffer,
        op
    )


@register_impl(dt.logical._LogicalNot)
def logical_not(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        '!(bool)x'
    )
