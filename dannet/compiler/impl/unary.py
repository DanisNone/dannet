from functools import partial
from typing import Callable, Literal
import dannet as dt
from dannet.gradient import GradientOp
from dannet.core import Tensor
from dannet.dtypes import DannetDtype
from dannet.typing import DTypeLike, DTypeLikeO
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info,
)


def compile_unary(
    device: dt.Device,
    inputs: tuple[DannetDtype],
    output: DannetDtype,
    op_name: str,
    work_dtype: Literal["A", "B"] = "B",
) -> dt.device.DeviceKernel:
    dtypeA, = inputs
    dtypeB = output

    work = dtypeA if work_dtype == "A" else dtypeB

    op = (f"""
dt_$dtypeB$ operation(dt_$dtypeA$ x_inp)
{{
    dt_$work$ x = dt_convert_$dtypeA$_to_$work$(x_inp);
    return dt_{op_name}_$work$(x);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        work=work,
    )
    build_info.add_header(op)

    return build_program(device, "unary.cl", build_info).unary


dtype_func_type = Callable[
    [DTypeLike, DTypeLikeO],
    DannetDtype
]


def make_unary(
    op_name: str,
    result_dtype: dtype_func_type,
    work_dtype: Literal["A", "B"] = "B"
) -> Callable[[Tensor, DTypeLikeO], Tensor]:
    def inner(
        x: Tensor,
        /,
        dtype: DTypeLikeO = None
    ) -> Tensor:
        device = dt.current_device()

        if dtype is not None:
            dtype = dt.dtypes.normalize_dtype(dtype)

        dt.utils.check_device(op_name, "x", x, device)
        y = dt.empty_like(x, dtype=result_dtype(x.dtype, dtype), device=device)

        kernel = compile_unary(
            device,
            inputs=(x.dtype, ),
            output=y.dtype,
            op_name=op_name,
            work_dtype=work_dtype
        )

        event = kernel(
            (x.size, ), None,
            x._buffer, y._buffer,
            get_shape_info(device, x),
            get_shape_info(device, y)
        )
        return Tensor(
            y._buffer, y._tensor_info,
            event=event
        )
    return inner


def not_bool(
    name: str,
    x: DTypeLike,
    dtype: DTypeLikeO
) -> DannetDtype:
    if dtype is not None:
        x = dtype
    x = dt.utils.normalize_dtype(x)
    if x == dt.bool_:
        raise ValueError(f"{name} not support bool dtype.")
    return x


def identity_dtype(x: DTypeLike, dtype: DTypeLikeO) -> DannetDtype:
    return dt.dtypes.normalize_dtype(dtype or x)


def to_inexact(
    x: DTypeLike,
    dtype: DTypeLikeO
) -> DannetDtype:
    if dtype is not None:
        x = dtype
    return dt.dtypes.promote_to_inexact(x)


gradop = partial(GradientOp, nondiff_argnum=(1,))

negative_op = make_unary("negative", partial(not_bool, "negative"))
negative = gradop(negative_op, lambda grad, out, args, kwargs: -grad)

positive_op = make_unary("positive", identity_dtype)
_positive = gradop(positive_op, lambda grad, out, args, kwargs: grad)


def positive(x: dt.typing.TensorLike, /, dtype: DTypeLikeO = None) -> Tensor:
    x = dt.array(x)
    if dtype is None:
        dtype = x.dtype
    dtype = dt.dtypes.normalize_dtype(dtype)
    if x.dtype == dtype:
        return x
    return _positive(x, dtype)


def astype(x: dt.typing.TensorLike, dtype: DTypeLike | None) -> Tensor:
    return positive(x, dtype)


def copy(x: dt.typing.TensorLike) -> Tensor:
    return _positive(x)


negative_op = make_unary("negative", partial(not_bool, "negative"))
negative = gradop(negative_op, lambda grad, out, args, kwargs: -grad)

square_op = make_unary("square", identity_dtype)
square = gradop(
    square_op,
    lambda grad, out, args, kwargs: 2 * grad * args[0]
)

sqrt_op = make_unary("sqrt", to_inexact)
sqrt = gradop(
    sqrt_op,
    lambda grad, out, args, kwargs: grad / (2 * out)
)

sin_op = make_unary("sin", to_inexact)
sin = gradop(
    sin_op,
    lambda grad, out, args, kwargs: grad * cos(args[0])  # type: ignore
)

cos_op = make_unary("cos", to_inexact)
cos = gradop(cos_op, lambda grad, out, args, kwargs: -grad * sin(args[0]))

tan_op = make_unary("tan", to_inexact)
tan = gradop(
    tan_op,
    lambda grad, out, args, kwargs: grad * (1 + dt.square(out))
)

sinh_op = make_unary("sinh", to_inexact)
sinh = gradop(
    sinh_op,
    lambda grad, out, args, kwargs: grad * cosh(args[0])   # type: ignore
)

cosh_op = make_unary("cosh", to_inexact)
cosh = gradop(
    cosh_op,
    lambda grad, out, args, kwargs: grad * sinh(args[0])
)

tanh_op = make_unary("tanh", to_inexact)
tanh = gradop(
    tanh_op,
    lambda grad, out, args, kwargs: grad * (1 - dt.square(out))
)

exp_op = make_unary("exp", to_inexact)
exp = gradop(exp_op, lambda grad, out, args, kwargs: grad * out)

log_op = make_unary("log", to_inexact)
log = gradop(log_op, lambda grad, out, args, kwargs: grad / args[0])
