from typing import Any, Callable, NoReturn
import dannet as dt
from dannet import dtypes
from dannet.gradient import GradientOp


def broadcast_shapes(*shapes: dt.typing.ShapeLike) -> tuple[int, ...]:
    norm_shapes = tuple(dt.utils.normalize_shape(s) for s in shapes)
    ndim = max(map(len, norm_shapes))

    result = [1] * ndim
    for shape in norm_shapes:
        shape = (1, ) * (ndim - len(shape)) + shape
        for i, (dim1, dim2) in enumerate(zip(result, shape)):
            if dim1 == dim2:
                continue
            elif dim1 == 1:
                result[i] = dim2
            else:
                raise ValueError(
                    f"fail broadcast shapes: {shapes=}"
                )
    return tuple(result)


def _not_implemented_grad(name: str) -> Callable[..., NoReturn]:
    def _(*args: Any, **kwargs: Any) -> NoReturn:
        raise NotImplementedError(f"gradient for {name} not implemented")
    return _


def _broadcast_to(
    x: dt.typing.TensorLike,
    shape: dt.typing.ShapeLike
) -> dt.core.Tensor:
    shape = dt.utils.normalize_shape(shape)

    x = dt.array(x, device=dt.current_device())
    if broadcast_shapes(x.shape, shape) != shape:
        raise ValueError(
            f"fail broadcast shape {x.shape} to {shape}"
        )

    pad_shape = (1, ) * (len(shape) - x.ndim) + x.shape
    new_strides = [0] * (len(shape) - x.ndim) + list(x.strides)

    for i in range(len(shape)):
        if pad_shape[i] != shape[i]:
            new_strides[i] = 0

    return dt.core.Tensor(
        x._buffer,
        dt.core.TensorInfo(shape, x.dtype, tuple(new_strides))
    )


broadcast_to = GradientOp(
    _broadcast_to,
    _not_implemented_grad('broadcast_to'),
    nondiff_argnum=(1,)
)


def full(
    shape: dt.typing.ShapeLike,
    fill_value: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None, *,
    device: dt.Device | None = None
) -> dt.core.Tensor:
    res = dt.array(fill_value, dtype=dtype, device=device)
    return dt.broadcast_to(res, shape)


def full_like(
    a: dt.typing.TensorLike,
    fill_value: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> dt.core.Tensor:
    a = dt.array(a, device=device)
    if dtype is None:
        dtype = a.dtype
    if shape is None:
        shape = a.shape
    return full(shape, fill_value, dtype, device=device)


def zeros(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike = dtypes.float64,
    *, device: dt.Device | None = None
) -> dt.core.Tensor:
    return full(shape, 0, dtype, device=device)


def zeros_like(
    a: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> dt.core.Tensor:
    return full_like(a, fill_value=0, dtype=dtype, shape=shape, device=device)


def ones(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike = dtypes.float64,
    *, device: dt.Device | None = None
) -> dt.core.Tensor:
    return full(shape, 1, dtype, device=device)


def ones_like(
    a: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    shape: dt.typing.ShapeLike | None = None, *,
    device: dt.Device | None = None
) -> dt.core.Tensor:
    return full_like(a, fill_value=1, dtype=dtype, shape=shape, device=device)


__all__ = [
    "broadcast_shapes", "broadcast_to",
    "full", "zeros", "ones",
    "full_like", "zeros_like", "ones_like"
]
