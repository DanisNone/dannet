import dannet as dt


def vector_norm(
        x: dt.typing.TensorLike, /, *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | str = 2
):
    arr = dt.convert_to_tensor(x)
    arr = arr.cast(dt.dtype.promote_to_float(arr.dtype))

    if ord is None or ord == 2:
        return dt.sqrt(
            dt.sum(
                dt.real(arr * dt.conj(arr)),
                axis=axis, keepdims=keepdims
            )
        )
    elif ord == dt.inf:
        return dt.max(dt.abs(arr), axis=axis, keepdims=keepdims)
    elif ord == -dt.inf:
        return dt.min(dt.abs(arr), axis=axis, keepdims=keepdims)
    elif ord == 0:
        return dt.sum(arr != 0, axis=axis, keepdims=keepdims, dtype=arr.dtype)
    elif ord == 1:
        return dt.sum(dt.abs(arr), axis=axis, keepdims=keepdims)
    elif isinstance(ord, str):
        msg = f'Invalid order \'{ord}\' for vector norm.'
        if ord == 'inf':
            msg += 'Use \'dt.inf\' instead.'
        if ord == '-inf':
            msg += 'Use \'-dt.inf\' instead.'
        raise ValueError(msg)
    else:
        abs_x = dt.abs(arr)
        ord_arr = dt.cast(ord, abs_x.dtype)
        ord_inv = dt.cast(1 / ord_arr, abs_x.dtype)
        out = dt.sum(dt.power(abs_x, ord_arr), axis=axis, keepdims=keepdims)
        return dt.power(out, ord_inv)


def norm(
    x: dt.typing.TensorLike,
    ord: int | str | None = None,
    axis: None | tuple[int, ...] | int = None,
    keepdims: bool = False
):
    x_arr = dt.convert_to_tensor(x)
    x_arr = x_arr.cast(dt.dtype.promote_to_float(x_arr.dtype))

    if axis is None:
        if ord is None:
            return dt.sqrt(
                dt.sum(
                    dt.real(x_arr * dt.conj(x_arr)),
                    keepdims=keepdims
                )
            )
        axis = tuple(range(x_arr.ndim))
    elif isinstance(axis, tuple):
        axis = tuple(
            dt.utils.normalize_axis_index(a, x_arr.ndim) for a in axis
        )
    else:
        axis = (dt.utils.normalize_axis_index(axis, x_arr.ndim),)

    num_axes = len(axis)
    if num_axes == 1:
        return vector_norm(
            x_arr,
            ord=2 if ord is None else ord,
            axis=axis, keepdims=keepdims
        )

    elif num_axes == 2:
        row_axis, col_axis = axis  # type: ignore
        if ord is None or ord in ('f', 'fro'):
            return dt.sqrt(
                dt.sum(
                    dt.real(x_arr * dt.conj(x_arr)),
                    axis=axis, keepdims=keepdims)
               )
        elif ord == 1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            return dt.max(
                dt.sum(dt.abs(x_arr), axis=row_axis, keepdims=keepdims),
                axis=col_axis, keepdims=keepdims
            )
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            return dt.min(
                dt.sum(dt.abs(x_arr), axis=row_axis, keepdims=keepdims),
                axis=col_axis, keepdims=keepdims
            )
        elif ord == dt.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            return dt.max(
                dt.sum(dt.abs(x_arr), axis=col_axis, keepdims=keepdims),
                axis=row_axis, keepdims=keepdims
            )
        elif ord == -dt.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            return dt.min(
                dt.sum(dt.abs(x_arr), axis=col_axis, keepdims=keepdims),
                axis=row_axis, keepdims=keepdims
            )
        elif ord in ('nuc', 2, -2):
            raise NotImplementedError(f'ord = \'{ord}\' not implemented')
        else:
            raise ValueError(f'Invalid order \'{ord}\' for matrix norm.')
    else:
        raise ValueError(
            f'Improper number of axes for norm: {axis=}. Pass one axis to '
            'compute a vector-norm, or two axes to compute a matrix-norm.')


def matrix_norm(
    x: dt.typing.TensorLike, /, *,
    keepdims: bool = False, ord: str | int = 'fro'
):
    return norm(x, ord=ord, keepdims=keepdims, axis=(-2, -1))
