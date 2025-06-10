import operator
from typing import SupportsIndex
import dannet as dt
from dannet.dtypes import normalize_dtype  # noqa: F401


def normalize_shape(shape: dt.typing.ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, SupportsIndex):
        return (operator.index(shape), )
    return tuple(operator.index(dim) for dim in shape)


def check_device(
    name: str,
    arg_name: str,
    tensor: dt.core.Tensor,
    device: dt.Device
) -> None:
    if tensor.device != device:
        # TODO: add message
        raise ValueError(f"{(name, arg_name, tensor.device, device)=}")

def normalize_axis_index(axis: SupportsIndex, ndim: int, argname: str | None = None):
    axis = operator.index(axis)
    if not (-ndim <= axis < ndim):
        if argname:
            argname = f"{argname}: "
        else:
            argname = ""
        raise ValueError(f"{argname} axis {axis} is out of bounds for array of dimension {ndim}")
    if axis < 0:
        axis += ndim
    return axis


def normalize_axis_tuple(axis: dt.typing.ShapeLike, ndim: int, argname: str | None = None):
    if isinstance(axis, SupportsIndex):
        axis = [axis]
    axis = tuple([normalize_axis_index(ax, ndim, argname) for ax in axis])
    if len(set(axis)) != len(axis):
        if argname:
            raise ValueError("repeated axis in `{}` argument".format(argname))
        else:
            raise ValueError("repeated axis")
    return axis
