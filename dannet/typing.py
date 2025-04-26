from typing import Any, Sequence, SupportsIndex

import numpy as np
import dannet as dt

ShapeLike = Sequence[SupportsIndex]
DTypeLike = np.dtype | str
TensorLike = dt.core.TensorBase | np.typing.NDArray[Any] | int | float