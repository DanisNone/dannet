import dannet as dt
from .utils import (
    Generator,
    register_impl,
    build_kernel
)


def unary(
    device,
    node: dt.math._ElementWiseUnary,
    input_buffers,
    output_buffer,
    op: str,
    workA: str = 'B',
    workB: str = 'B',
    custom: bool = False
):
    (A,) = input_buffers
    B = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.x, B=node)

    if custom:
        gen.line(f'''
dtypeB operation(dtypeA x)
{{
    {op}
}}
''')
    else:
        t = {'A': node.x.dtype, 'B': node.dtype}
        workA = t.get(workA, workA)
        workB = t.get(workB, workB)

        gen.line(f'''
dtypeB operation(dtypeA x_inp)
{{
    dt_workA x = dt_convert_dtypeA_to_workA(x_inp);
    return dt_{op}_workB(x);
}}
''')
        gen.dtype_names(workA=workA, workB=workB)

    if node.x._is_contiguous:
        gen.mode('full')
    else:
        gen.mode('strided')

    global_size = (node.size,)
    local_size = None
    kernel = build_kernel(device, 'unary.cl', gen)

    return lambda: kernel.general(device.queue, global_size, local_size, A, B)


def register_unary(
    class_: type[dt.ops.math._ElementWiseUnary],
    op: str, custom: bool = False
):
    @register_impl(class_)
    def inner(device, node, input_buffers, output_buffer):
        return unary(
            device, node, input_buffers,
            output_buffer, op, custom=custom
        )


@register_impl(dt.math._Abs)
def abs(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'abs', workA='A', workB='A'
    )


@register_impl(dt.logical._LogicalNot)
def logical_not(device, node, input_buffers, output_buffer):
    return unary(
        device, node, input_buffers, output_buffer,
        'logical_not', workA='A', workB='A'
    )


register_unary(dt.math._Negative, 'negative')
register_unary(dt.math._Square, 'square')

register_unary(dt.math._Exp, 'exp')
register_unary(dt.math._Exp2, 'exp2')
register_unary(dt.math._Exp10, 'exp10')
register_unary(dt.math._Expm1, 'expm1')

register_unary(dt.math._Log, 'log')
register_unary(dt.math._Log2, 'log2')
register_unary(dt.math._Log10, 'log10')
register_unary(dt.math._Log1p, 'log1p')

register_unary(dt.math._Sqrt, 'sqrt')
register_unary(dt.math._Rsqrt, 'rsqrt')

register_unary(dt.math._Sin, 'sin')
register_unary(dt.math._Cos, 'cos')
register_unary(dt.math._Tan, 'tan')

register_unary(dt.math._Sinh, 'sinh')
register_unary(dt.math._Cosh, 'cosh')
register_unary(dt.math._Tanh, 'tanh')

register_unary(dt.math._Arcsin, 'arcsin')
register_unary(dt.math._Arccos, 'arccos')
register_unary(dt.math._Arctan, 'arctan')

register_unary(dt.math._Arcsinh, 'arcsinh')
register_unary(dt.math._Arccosh, 'arccosh')
register_unary(dt.math._Arctanh, 'arctanh')

register_unary(dt.math._Sign, 'sign')


register_unary(dt.math._Round, 'round')
register_unary(dt.math._Trunc, 'trunc')
register_unary(dt.math._Floor, 'floor')
register_unary(dt.math._Ceil, 'ceil')

register_unary(dt.bitwise._BitwiseNot, 'bitwise_not')

register_unary(
    dt.nnet.activations._Relu,
    'return dt_max_dtypeB(dt_convert_dtypeA_to_dtypeB(x), dt_zero_dtypeB());',
    custom=True
)

register_unary(
    dt.nnet.activations._Relu6,
    '''
return dt_max_dtypeB(
    dt_min_dtypeB(
        dt_convert_dtypeA_to_dtypeB(x),
        dt_convert_int32_to_dtypeB(6)
    ),
    dt_zero_dtypeB()
);
''',
    custom=True
)

register_unary(
    dt.nnet.activations._Sigmoid,
    '''
return dt_divide_dtypeB(dt_one_dtypeB(),
    dt_add_dtypeB(dt_one_dtypeB(),
        dt_exp_dtypeB(dt_negative_dtypeB(
            dt_convert_dtypeA_to_dtypeB()
        ))
    )
);
''',
    custom=True
)

register_unary(
    dt.nnet.activations._Softplus,
    '''
return dt_log_dtypeB(
    dt_add_dtypeB(dt_one_dtypeB(),
        dt_exp_dtypeB(
            dt_convert_dtypeA_to_dtypeB()
        )
    )
);
''',
    custom=True
)

register_unary(
    dt.nnet.activations._Softsign,
    '''
dtypeB xn = dt_convert_dtypeA_to_dtypeB(x);
return dt_divide_dtypeB(x,
    dt_add_dtypeB(
        dt_one_dtypeB(),
        dt_abs_dtypeB(xn)
    )
);
''',
    custom=True
)


register_unary(
    dt.nnet.activations._Softsign,
    '''

const dtypeB c3 = dt_convert_uint32_to_dtypeB(3);
const dtypeB c6 = dt_convert_uint32_to_dtypeB(6);
const dtypeB cn3 = dt_convert_uint32_to_dtypeB(-3);
dtypeB xn = dt_convert_dtypeA_to_dtypeB(x);

if (dt_less_dtypeB(xn, cn3))
    return dt_zero_dtypeB();
if (dt_greater_dtypeB(xn, c3))
    return dt_one_dtypeB();

return dt_divide_dtypeB(dt_add_dtypeB(x, c3), c6);
''',
    custom=True
)
