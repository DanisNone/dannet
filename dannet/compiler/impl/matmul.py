import math
import dannet as dt
from dannet.core import Tensor
from dannet.dtypes import DannetDtype
from dannet.typing import DTypeLikeO
from dannet.compiler.impl.utils import (
    build_program,
    BuildInfo,
    get_shape_info,
)


def compile_matmul(
    device: dt.Device,
    inputs: tuple[DannetDtype, DannetDtype],
    output: DannetDtype,
) -> dt.device.DeviceKernel:
    dtypeA, dtypeB = inputs
    dtypeC = output

    build_info = BuildInfo()
    build_info.add_dtypes(
        dtypeA=dtypeA,
        dtypeB=dtypeB,
        dtypeC=dtypeC,
    )
    return build_program(device, "matmul.cl", build_info).matmul


def matmul(x1: Tensor, x2: Tensor, /, dtype: DTypeLikeO = None) -> Tensor:
    if dtype is not None:
        raise NotImplementedError
    device = dt.current_device()

    if x1.ndim < 2 or x2.ndim < 2:
        raise ValueError("matmul with ndim < 2 not implemented.")

    dt.utils.check_device("matmul", "x1", x1, device)
    dt.utils.check_device("matmul", "x2", x2, device)

    batch1, batch2 = x1.shape[:-2], x2.shape[:-2]

    M, N_ = x1.shape[-2:]
    N, K = x2.shape[-2:]

    if N != N_:
        # TODO: add message
        raise ValueError
    if x1.ndim != x2.ndim:
        raise NotImplementedError
    batch = dt.broadcast_shapes(batch1, batch2)
    shape = batch + (M, K)
    dtype = dt.promote_types(x1.dtype, x2.dtype)
    out = dt.empty(shape, dtype=dtype, device=device)

    kernel = compile_matmul(
        device,
        inputs=(x1.dtype, x2.dtype),
        output=out.dtype
    )

    # TODO: implement tile-matmul
    event = kernel(
        (math.prod(batch), M, K), None,
        x1._buffer, x2._buffer, out._buffer,
        get_shape_info(device, x1),
        get_shape_info(device, x2),
        get_shape_info(device, out),
    )
    return Tensor(
        out._buffer, out._tensor_info,
        event=event
    )
