import dannet as dt
from .utils import (
    Generator,
    build_kernel,
    register_impl
)


def ternary(
    device,
    node: dt.math._ElementWiseTernary,
    input_buffers,
    output_buffer,
    op: str,
    workA: str = 'D',
    workB: str = 'D',
    workC: str = 'D',
    workD: str = 'D'
):
    assert node._is_contiguous

    A, B, C = input_buffers
    D = output_buffer

    gen = Generator()
    gen.nodes_info(A=node.x, B=node.y, C=node.z, D=node)

    t = {
        'A': node.x.dtype,
        'B': node.y.dtype,
        'C': node.z.dtype,
        'D': node.dtype
    }
    workA = t.get(workA, workA)
    workB = t.get(workB, workB)
    workC = t.get(workC, workC)
    workD = t.get(workD, workD)

    gen.line(f'''
dtypeC operation(dtypeA x, dtypeB y, dtypeC z) {{
{op}
}}''')
    gen.dtype_names(workA=workA, workB=workB, workC=workC)
    gen.mode('strided')

    global_size = (node.size,)
    local_size = None

    kernel = build_kernel(device, 'ternary.cl', gen)
    return lambda: kernel.general(
        device.queue, global_size, local_size,
        A, B, C, D
    )


@register_impl(dt.math._Where)
def where(device, node, input_buffers, output_buffer):
    return ternary(
        device, node, input_buffers, output_buffer,
        '''
    dt_bool cond = dt_convert_dtypeA_to_bool(x);
    return (
        cond ?
        dt_convert_dtypeB_to_dtypeD(y) :
        dt_convert_dtypeC_to_dtypeD(z)
    );
''')


@register_impl(dt.math._Clip)
def clip(device, node, input_buffers, output_buffer):
    return ternary(
        device, node, input_buffers, output_buffer,
        '''
    dtypeD xn = dt_convert_dtypeA_to_dtypeD(x);
    dtypeD yn = dt_convert_dtypeB_to_dtypeD(y);
    dtypeD zn = dt_convert_dtypeC_to_dtypeD(z);

    return dt_min_dtypeD(dt_max_dtypeD(xn, yn), zn);
'''
    )
