from functools import partial
from typing import Any, Callable
import dannet as dt
from dannet.gradient import GradientOp
from dannet.core import Tensor
from dannet.dtypes import DannetDtype
from dannet.typing import DTypeLike, DTypeLikeO

from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info
)


def compile_binary(
    device: dt.Device,
    inputs: tuple[DannetDtype, DannetDtype],
    output: DannetDtype,
    op_name: str,
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    op = (f"""
dt_$dtypeC$ operation(dt_$dtypeA$ x_inp, dt_$dtypeB$ y_inp)
{{
    dt_$dtypeC$ x = dt_convert_$dtypeA$_to_$dtypeC$(x_inp);
    dt_$dtypeC$ y = dt_convert_$dtypeB$_to_$dtypeC$(y_inp);
    return dt_{op_name}_$dtypeC$(x, y);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC
    )
    build_info.add_header(op)
    return build_program(device, "binary.cl", build_info).binary


dtype_func_type = Callable[
    [DTypeLike, DTypeLike, DTypeLikeO],
    DannetDtype
]


def make_binary(
    op_name: str,
    result_dtype: dtype_func_type
) -> Callable[[Tensor, Tensor, DTypeLikeO], Tensor]:
    def inner(
        x1: Tensor,
        x2: Tensor,
        /,
        dtype: DTypeLikeO = None
    ) -> Tensor:
        device = dt.current_device()

        if dtype is not None:
            dtype = dt.dtypes.normalize_dtype(dtype)
        dt.utils.check_device(op_name, "x1", x1, device)
        dt.utils.check_device(op_name, "x2", x2, device)

        if x1.shape != x2.shape:
            raise NotImplementedError(f"{(x1.shape, x2.shape)=}")
        out = dt.empty(
            x1.shape,
            dtype=result_dtype(x1.dtype, x2.dtype, dtype),
            device=device
        )

        kernel = compile_binary(
            device,
            inputs=(x1.dtype, x2.dtype),
            output=out.dtype,
            op_name=op_name
        )

        event = kernel(
            (out.size, ), None,
            x1._buffer, x2._buffer, out._buffer,
            get_shape_info(device, x1),
            get_shape_info(device, x2),
            get_shape_info(device, out)
        )
        return Tensor(
            out._buffer, out._tensor_info,
            event=event
        )
    return inner


def to_inexact(
    name: str,
    x1: DTypeLike, x2: DTypeLike,
    dtype: DTypeLikeO
) -> DannetDtype:
    if dtype is not None:
        out = dt.dtypes.normalize_dtype(dtype)
    else:
        out = dt.promote_types(x1, x2)
    if not dt.dtypes.is_inexact_dtype(out):
        # TODO: add message
        raise ValueError("")
    return out


def promote(x1: DTypeLike, x2: DTypeLike, dtype: DTypeLikeO) -> DannetDtype:
    if dtype is not None:
        return dt.dtypes.normalize_dtype(dtype)
    return dt.promote_types(x1, x2)


add_op = make_binary("add", promote)
add = GradientOp(add_op, lambda grad, out, args, kwargs: (grad, grad))

subtract_op = make_binary("subtract", promote)
subtract = GradientOp(
    subtract_op,
    lambda grad, out, args, kwargs: (grad, -grad)
)

multiply_op = make_binary("multiply", promote)
multiply = GradientOp(
    multiply_op,
    lambda grad, out, args, kwargs: (grad * args[1], grad * args[0])
)

divide_op = make_binary("divide", partial(to_inexact, "divide"))
divide = GradientOp(
    divide_op,
    lambda grad, out, args, kwargs: (grad / args[1], -grad*out/args[1])
)


def power_dtype(
    x1: DTypeLike, x2: DTypeLike,
    dtype: DTypeLikeO
) -> DannetDtype:
    if dt.dtypes.is_bool_dtype(x1) and not dt.dtypes.is_inexact_dtype(x2):
        out = dt.int32
    elif dt.dtypes.is_inexact_dtype(x1) or dt.dtypes.is_inexact_dtype(x2):
        out = dt.promote_types(x1, x2)
    else:
        out = dt.dtypes.normalize_dtype(x1)

    if dtype is not None:
        # TODO: add message
        raise NotImplementedError("")
    return out


def power_grad(
    grad: Tensor, out: Tensor,
    args: tuple[Tensor, Tensor], kwargs: Any
) -> tuple[Tensor, Tensor]:
    x1, x2 = args
    grad_x = grad * x2 * x1 ** (x2 - dt.ones_like(x2))
    grad_y = grad * out * dt.log(x1)
    return (grad_x, grad_y)


power_op = make_binary("power", power_dtype)
power = GradientOp(power_op, power_grad)
