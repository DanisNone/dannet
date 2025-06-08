from typing import Callable, Literal
import dannet as dt
from dannet.compiler.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_unary(
    device: dt.Device,
    inputs: list[dt.dtypes.DannetDtype],
    output: dt.dtypes.DannetDtype,
    op_name: str,
    cast_x_to: Literal["A", "B"] = "B",
    work: Literal["A", "B"] = "B",
) -> dt.device.DeviceKernel:
    dtypeA, = inputs
    dtypeB = output

    workA = dtypeA if cast_x_to == "A" else dtypeB
    workB = dtypeA if work == "A" else dtypeB

    op = (f"""
$dtypeB$ operation($dtypeA$ x_inp)
{{
    $dt_workA$ x = dt_convert_$dtypeA$_to_$workA$(x_inp);
    return dt_{op_name}_$workB$(x);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        workA=workA,
        workB=workB,
    )
    build_info.add_header(op)

    return build_program(device, "unary.cl", build_info).unary


dtype_func_type = Callable[
    [dt.typing.DtypeLike, dt.typing.DtypeLikeO],
    dt.dtypes.DannetDtype
]


def make_unary(
    op_name: str,
    result_dtype: dtype_func_type,
    cast_x_to: Literal["A", "B"] = "B",
    work: Literal["A", "B"] = "B",
) -> Callable[[dt.typing.TensorLike], dt.core.Tensor]:
    def inner(
        x: dt.typing.TensorLike,
        dtype: dt.typing.DtypeLikeO = None
    ) -> dt.core.Tensor:
        device = dt.current_device()

        x = dt.array(x, device=device)
        y = dt.empty_like(x, dtype=result_dtype(x.dtype, dtype), device=device)

        kernel = compile_unary(
            device,
            inputs=[x.dtype],
            output=y.dtype,
            op_name=op_name,
            cast_x_to=cast_x_to,
            work=work
        )

        event = kernel(
            (x.size, ), None,
            x, y,
            get_shape_info(device, x),
            get_shape_info(device, y)
        )
        return dt.core.Tensor(
            y._buffer, y._tensor_info,
            event=event
        )
    return inner


def negative_result(
    x: dt.typing.DtypeLike,
    dtype: dt.typing.DtypeLikeO
) -> dt.dtypes.DannetDtype:
    x = dt.utils.normalize_dtype(x)
    if dtype is not None:
        x = dt.utils.normalize_dtype(dtype)
    if x == dt.bool_:
        raise ValueError
    return x


negative = make_unary("negative", negative_result)
