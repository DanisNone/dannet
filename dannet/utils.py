import operator
from typing import SupportsIndex
import dannet as dt
from dannet.dtypes import normalize_dtype  # noqa: F401


def normalize_shape(shape: dt.typing.ShapeLike) -> tuple[int, ...]:
    if isinstance(shape, SupportsIndex):
        return (operator.index(shape), )
    return tuple(operator.index(dim) for dim in shape)
