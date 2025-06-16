import operator
from typing import SupportsIndex
from dannet.lib.dtypes import normalize_dtype  # noqa: F401
from dannet.lib.typing import ShapeLike


def normalize_shape(shape: ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, SupportsIndex):
        return (operator.index(shape), )
    return tuple(operator.index(dim) for dim in shape)


def normalize_axis_index(
    axis: SupportsIndex,
    ndim: int, argname: str | None = None
) -> int:
    axis = operator.index(axis)
    if not (-ndim <= axis < ndim):
        if argname:
            argname = f"{argname}: "
        else:
            argname = ""
        raise ValueError(
            f"{argname} axis {axis} is out of bounds "
            f"for array of dimension {ndim}"
        )
    if axis < 0:
        axis += ndim
    return axis


def normalize_axis_tuple(
    axis: ShapeLike,
    ndim: int, argname: str | None = None
) -> tuple[int, ...]:
    if isinstance(axis, SupportsIndex):
        axis = [axis]
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if len(set(axis)) != len(axis):
        if argname:
            raise ValueError("repeated axis in `{}` argument".format(argname))
        else:
            raise ValueError("repeated axis")
    return axis
