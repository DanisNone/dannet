import dannet as dt


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


def broadcast_to(x: dt.typing.TensorLike, shape: dt.typing.ShapeLike) -> dt.core.Tensor:
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


__all__ = ["broadcast_shapes", "broadcast_to"]
