from typing import Any, Sequence, SupportsIndex

import numpy as np
import dannet as dt

ShapeLike = Sequence[SupportsIndex] | SupportsIndex
DTypeLike = np.dtype | str | type[bool] | type[int] | type[float]
DTypeLikeO = DTypeLike | None

TensorLike = (
    dt.core.TensorBase |
    np.typing.NDArray[Any]
    | int | float | list |
    np.generic
)


Axis = Sequence[SupportsIndex] | SupportsIndex
AxisO = Axis | None
