from functools import partial
from typing import Any, Callable, Literal
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
    cast_to: Literal["out", "in"] = "out"
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    if cast_to == "in":
        if dtypeA != dtypeB:
            # TODO: add message
            raise ValueError()
        work = dtypeA
    else:
        work = dtypeC
    
    op = (f"""
dt_$dtypeC$ operation(dt_$dtypeA$ x_inp, dt_$dtypeB$ y_inp)
{{
    dt_$work$ x = dt_convert_$dtypeA$_to_$work$(x_inp);
    dt_$work$ y = dt_convert_$dtypeB$_to_$work$(y_inp);
    return dt_{op_name}_$work$(x, y);
}}
""")

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
        work=work
    )
    build_info.add_header(op)
    return build_program(device, "binary.cl", build_info).binary


dtype_func_type = Callable[
    [DTypeLike, DTypeLike, DTypeLikeO],
    DannetDtype
]


def make_binary(
    op_name: str,
    result_dtype: dtype_func_type,
    cast_to: Literal["in", "out"] = "out"
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
            op_name=op_name,
            cast_to=cast_to
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


def require_bool(x1: DTypeLike, x2: DTypeLike, dtype: DTypeLikeO) -> DannetDtype:
    if dtype is not None:
        dtype = dt.dtypes.normalize_dtype(dtype)
        if dtype != dt.bool_:
            # TODO: add message
            raise ValueError()
    return dt.bool_


def gradop(fwd, bwd):
    def fwd_new(
        x1: Tensor,
        x2: Tensor,
        /,
        dtype: DTypeLikeO = None
    ) -> Tensor:
        shape = dt.broadcast_shapes(x1.shape, x2.shape)
        if x1.shape != shape:
            x1 = dt.broadcast_to(x1, shape)
        if x2.shape != shape:
            x2 = dt.broadcast_to(x2, shape)
        return fwd(x1, x2, dtype)
    
    def bwd_new(
        grad: Tensor, out: Tensor,
        args: tuple[Tensor, Tensor, DTypeLikeO], kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        g1, g2 = bwd(grad, out, args, kwargs)
        if g1.shape != args[0].shape:
            g1 = dt.reduce_to(g2, args[0].shape)
        if g2.shape != args[0].shape:
            g2 = dt.reduce_to(g2, args[0].shape)
        return (g1, g2)
    return GradientOp(fwd_new, bwd_new, nondiff_argnum=(2,))

add_op = make_binary("add", promote)
add = gradop(add_op, lambda grad, out, args, kwargs: (grad, grad))

subtract_op = make_binary("subtract", promote)
subtract = gradop(
    subtract_op,
    lambda grad, out, args, kwargs: (grad, -grad)
)

multiply_op = make_binary("multiply", promote)
multiply = gradop(
    multiply_op,
    lambda grad, out, args, kwargs: (grad * args[1], grad * args[0])
)

divide_op = make_binary("divide", partial(to_inexact, "divide"))
divide = gradop(
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
    args: tuple[Tensor, Tensor, DTypeLikeO], kwargs: Any
) -> tuple[Tensor, Tensor]:
    x1, x2 = args[:2]
    grad_x = grad * x2 * x1 ** (x2 - dt.ones_like(x2))
    grad_y = grad * out * dt.log(x1)
    return (grad_x, grad_y)


power_op = make_binary("power", power_dtype)
power = gradop(power_op, power_grad)


def make_cmp(op_name) -> GradientOp:
    cmp_op = make_binary(op_name, require_bool, cast_to="in")
    def fwd(x1: Tensor, x2: Tensor, /, dtype: DTypeLikeO = None) -> Tensor:
        dtype_ = dt.promote_types(x1.dtype, x2.dtype)
        return cmp_op(x1.astype(dtype_), x2.astype(dtype_), dtype)
    return GradientOp(fwd, lambda grad, out, args, kwargs: (None, None))

equal = make_cmp("equal")
not_equal = make_cmp("not_equal")
less = make_cmp("less")
less_equal = make_cmp("less_equal")
greater = make_cmp("greater")
greater_equal = make_cmp("greater_equal")
