from typing import Sequence, SupportsIndex
import numpy as np
import dannet as dt


_Scalar = bool | int | float | complex
TensorLike = dt.core.Tensor | _Scalar | Sequence[_Scalar] | np.typing.NDArray
DTypeLike = dt.dtypes.DannetDtype | np.dtype | str
DTypeLikeO = dt.dtypes.DannetDtype | np.dtype | str | None
ShapeLike = Sequence[SupportsIndex] | SupportsIndex
