from typing import Any, Sequence, SupportsIndex

import numpy as np
import dannet as dt

ShapeLike = Sequence[SupportsIndex]
DTypeLike = np.dtype | str | type[bool] | type[int] | type[float]
TensorLike = (
    dt.core.TensorBase |
    np.typing.NDArray[Any]
    | int | float | list |
    np.generic
)
