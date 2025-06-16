from typing import Sequence, SupportsIndex, TypeAlias

from dannet.lib.core import TensorLike  # noqa: F401
from dannet.lib.dtypes import DTypeLike  # noqa: F401

Axis: TypeAlias = Sequence[SupportsIndex] | SupportsIndex | None
ShapeLike: TypeAlias = Sequence[SupportsIndex] | SupportsIndex
