from .utils import *

def unary(device, node: dt.math._ElementWiseUnary, input_buffers, output_buffer, op: str):
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
    if node.x._is_default_strides():
        headers.append(generate_mode('full'))
    else:
        headers.append(generate_mode('strided'))
        
    global_size = (node.size,)
    local_size = None
    kernel = build_kernel(device, 'unary.cl', headers)

    return lambda: kernel.general(device.queue, global_size, local_size, A, B)


@register_impl(dt.math._Negative)
def negative(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, f'-x')

@register_impl(dt.math._Reciprocal)
def reciprocal(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, f'1 / x')

@register_impl(dt.math._Square)
def square(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x * x')


@register_impl(dt.math._Abs)
def abs(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x >= (dtypeA)0? x : -x')


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
    return unary(device, node, input_buffers, output_buffer, 'exp((dtypeB)x)')


@register_impl(dt.math._Log)
def log(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'log((dtypeB)x)')


@register_impl(dt.math._Sqrt)
def sqrt(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'sqrt((dtypeB)x)')

@register_impl(dt.math._Rsqrt)
def rsqrt(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, '1.0 / sqrt((dtypeB)x)')


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

@register_impl(dt.nnet.activations._Relu)
def relu(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x > 0 ? x : 0')

@register_impl(dt.nnet.activations._Relu6)
def relu6(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'clamp((dtypeB)x, (dtypeB)0, (dtypeB)6)')

@register_impl(dt.nnet.activations._Sigmoid)
def sigmoid(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, '1.0 / (1 + exp(-(dtypeB)x))')

@register_impl(dt.nnet.activations._Softplus)
def softplus(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'log(1 + exp((dtypeB)x))')

@register_impl(dt.nnet.activations._Softsign)
def softsign(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x / (1.0 + (dtypeB)(x < 0?-x:x))')

@register_impl(dt.nnet.activations._HardSigmoid)
def hard_sigmoid(device, node, input_buffers, output_buffer):
    return unary(device, node, input_buffers, output_buffer, 'x < -3 ? 0 : (x > 3? 1 : (dtypeB)x * (1.0 / 6.0) + 0.5)')
