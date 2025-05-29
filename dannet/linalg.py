import dannet as dt


def vector_norm(
        x: dt.typing.TensorLike, /, *,
        axis: int | tuple[int, ...] | None = None,
        keepdims: bool = False,
        ord: int | str = 2
):
    x = dt.convert_to_tensor(x)
    x = x.cast(dt.dtype.promote_to_float(x.dtype))

    if ord is None or ord == 2:
        return dt.sqrt(
            dt.sum(
                dt.real(x * dt.conj(x)),
                axis=axis, keepdims=keepdims
            )
        )
    elif ord == dt.inf:
        return dt.max(dt.abs(x), axis=axis, keepdims=keepdims)
    elif ord == -dt.inf:
        return dt.min(dt.abs(x), axis=axis, keepdims=keepdims)
    elif ord == 0:
        return dt.sum(x != 0, axis=axis, keepdims=keepdims).cast(x.dtype)
    elif ord == 1:
        return dt.sum(dt.abs(x), axis=axis, keepdims=keepdims)
    elif isinstance(ord, str):
        msg = f'Invalid order \'{ord}\' for vector norm.'
        if ord == 'inf':
            msg += 'Use \'dt.inf\' instead.'
        if ord == '-inf':
            msg += 'Use \'-dt.inf\' instead.'
        raise ValueError(msg)
    else:
        abs_x = dt.abs(x)
        ord_arr = dt.convert_to_tensor(ord).cast(abs_x.dtype)
        ord_inv = dt.convert_to_tensor(1. / ord_arr).cast(abs_x.dtype)
        out = dt.sum(dt.power(abs_x, ord_arr), axis=axis, keepdims=keepdims)
        return dt.power(out, ord_inv)


def norm(
    x: dt.typing.TensorLike,
    ord: int | str | None = None,
    axis: None | tuple[int, ...] | int = None,
    keepdims: bool = False
):
    x = dt.convert_to_tensor(x)
    x = x.cast(dt.dtype.promote_to_float(x.dtype))

    ndim = x.ndim

    if axis is None:
        if ord is None:
            return dt.sqrt(dt.sum(dt.real(x * dt.conj(x)), keepdims=keepdims))
        axis = tuple(range(ndim))
    elif isinstance(axis, tuple):
        axis = tuple(dt.utils.normalize_axis_index(x, ndim) for x in axis)
    else:
        axis = (dt.utils.normalize_axis_index(axis, ndim),)

    num_axes = len(axis)
    if num_axes == 1:
        return vector_norm(
            x,
            ord=2 if ord is None else ord,
            axis=axis, keepdims=keepdims
        )

    elif num_axes == 2:
        row_axis, col_axis = axis  # type: ignore
        if ord is None or ord in ('f', 'fro'):
            return dt.sqrt(
                dt.sum(
                    dt.real(x * dt.conj(x)),
                    axis=axis, keepdims=keepdims)
               )
        elif ord == 1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            return dt.max(
                dt.sum(dt.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis, keepdims=keepdims
            )
        elif ord == -1:
            if not keepdims and col_axis > row_axis:
                col_axis -= 1
            return dt.min(
                dt.sum(dt.abs(x), axis=row_axis, keepdims=keepdims),
                axis=col_axis, keepdims=keepdims
            )
        elif ord == dt.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            return dt.max(
                dt.sum(dt.abs(x), axis=col_axis, keepdims=keepdims),
                axis=row_axis, keepdims=keepdims
            )
        elif ord == -dt.inf:
            if not keepdims and row_axis > col_axis:
                row_axis -= 1
            return dt.min(
                dt.sum(dt.abs(x), axis=col_axis, keepdims=keepdims),
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
