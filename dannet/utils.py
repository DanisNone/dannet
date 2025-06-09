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
