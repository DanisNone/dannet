import dannet as dt
from .utils import (
    Generator,
    build_kernel,
    register_impl
)


def binary(
    device,
    node: dt.math._ElementWiseBinary,
    input_buffers,
    output_buffer,
    op: str,
    workA: str = 'C',
    workB: str = 'C',
    workC: str = 'C'
):
    assert node._is_contiguous

    A, B = input_buffers
    C = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.x, B=node.y, C=node)

    t = {'A': node.x.dtype, 'B': node.y.dtype, 'C': node.dtype}
    workA = t.get(workA, workA)
    workB = t.get(workB, workB)
    workC = t.get(workC, workC)

    gen.line(f'''
dtypeC operation(dtypeA x_inp, dtypeB y_inp) {{
    dt_workA x = dt_convert_dtypeA_to_workA(x_inp);
    dt_workB y = dt_convert_dtypeB_to_workB(y_inp);
    return dt_{op}_workC(x, y);
}}''')
    gen.dtype_names(workA=workA, workB=workB, workC=workC)

    if node.x._is_contiguous and node.y._is_contiguous:
        gen.mode('full')
    else:
        gen.mode('strided')

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'binary.cl', gen)
    return lambda: kernel.binary(
        device.queue, global_size, local_size,
        A, B, C
    )


def register_binary_simple(class_, op: str):
    @register_impl(class_)
    def inner(device, node, input_buffers, output_buffer):
        return binary(device, node, input_buffers, output_buffer, op)


def register_binary_cmp(class_, op: str):
    @register_impl(class_)
    def inner(device, node, input_buffers, output_buffer):
        dtype = dt.dtype.promote_dtypes(node.x.dtype, node.y.dtype)
        return binary(
            device, node, input_buffers, output_buffer,
            op, workA=dtype, workB=dtype, workC=dtype
        )


register_binary_simple(dt.math._Add, 'add')
register_binary_simple(dt.math._Subtract, 'subtract')
register_binary_simple(dt.math._Multiply, 'multiply')

register_binary_simple(dt.math._Divide, 'divide')
register_binary_simple(dt.math._FloorDivide, 'floor_divide')

register_binary_simple(dt.math._Power, 'power')

register_binary_simple(dt.math._Minimum, 'min')
register_binary_simple(dt.math._Maximum, 'max')

register_binary_cmp(dt.logical._Equal, 'equal')
register_binary_cmp(dt.logical._NotEqual, 'not_equal')

register_binary_cmp(dt.logical._Less, 'less')
register_binary_cmp(dt.logical._LessEqual, 'less_equal')

register_binary_cmp(dt.logical._Greater, 'greater')
register_binary_cmp(dt.logical._GreaterEqual, 'greater_equal')

register_binary_cmp(dt.logical._LogicalOr, 'logical_or')
register_binary_cmp(dt.logical._LogicalAnd, 'logical_and')
register_binary_cmp(dt.logical._LogicalXor, 'logical_xor')

register_binary_simple(dt.bitwise._BitwiseOr, 'bitwise_or')
register_binary_simple(dt.bitwise._BitwiseAnd, 'bitwise_and')
register_binary_simple(dt.bitwise._BitwiseXor, 'bitwise_xor')

register_binary_simple(dt.bitwise._LeftShift, 'left_shift')
register_binary_simple(dt.bitwise._RightShift, 'right_shift')

register_binary_simple(dt.math._Arctan2, 'arctan2')

register_binary_simple(dt.math._Logaddexp, 'logaddexp')
register_binary_simple(dt.math._Logaddexp2, 'logaddexp2')
