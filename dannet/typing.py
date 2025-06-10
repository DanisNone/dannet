from typing import Sequence, SupportsIndex, TypeAlias
import numpy as np
import dannet as dt


_Scalar: TypeAlias = bool | int | float | complex
TensorLike: TypeAlias = dt.core.Tensor | _Scalar | Sequence[_Scalar] | np.typing.NDArray

_ScalarType: TypeAlias = type[bool] | type[int] | type[float] | type[complex]
DTypeLike: TypeAlias = dt.dtypes.DannetDtype | np.dtype | str  | _ScalarType
DTypeLikeO: TypeAlias = DTypeLike | None
ShapeLike: TypeAlias = Sequence[SupportsIndex] | SupportsIndex
