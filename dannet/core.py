from __future__ import annotations
from typing import Sequence
import numpy as np

import dannet as dt


class TensorInfo:
    def __init__(
        self,
        shape: Sequence[int],
        dtype: dt.dtypes.DannetDtype,
        strides: tuple[int, ...] | None = None,
        buffer_offset: int = 0
    ):
        if min(shape, default=1) < 0:
            raise ValueError(
                f'All dims of Tensor must be non negative: {shape=}'
            )

        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = dt.utils.normalize_dtype(dtype)

        if strides is None:
            strides = self.get_strides(shape)
        self._strides = strides

        if buffer_offset < 0:
            raise ValueError(
                f'buffer offset must be non negative: {buffer_offset=}'
            )
        self._buffer_offset = buffer_offset

        if len(self._shape) > 64:
            raise ValueError(
                "maximum supported dimension for an Tensor "
                f"is currently 64. ndim={len(self._shape)}"
            )

    @staticmethod
    def get_strides(shape: Sequence[int]) -> tuple[int, ...]:
        strides: list[int] = []
        s = 1
        for dim in shape[::-1]:
            strides.append(s)
            s *= dim
        strides.reverse()
        return tuple(strides)

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> TensorInfo:
        strides = cls.get_strides(array.shape)
        return cls(
            shape=array.shape,
            dtype=array.dtype,
            strides=strides,
            buffer_offset=0
        )


class Tensor:
    def __init__(
        self,
        buffer: dt.device.DeviceBuffer,
        tensor_info: TensorInfo,
        event: dt.device.DeviceEvent | None = None,
    ):
        self._buffer = buffer
        self._tensor_info = tensor_info

        if event is None:
            event = dt.device.empty_event()
        self._event = event
        self._computed: np.ndarray | None = None

    def wait_compute(self) -> None:
        self._event.wait()

    def _read_from_buffer(self) -> np.ndarray:
        if self._computed is not None:
            return self._computed
        self.wait_compute()

        itemsize = dt.dtypes.itemsize(self._tensor_info._dtype)
        assert self._buffer.nbytes % itemsize == 0

        array = np.empty(
            self._buffer.nbytes // itemsize,
            dtype=self._tensor_info._dtype
        )
        dt.device.read_buffer(self._buffer, array)

        array_with_offset = array[self._tensor_info._buffer_offset:].copy()

        strides = [
            s * itemsize
            for s in self._tensor_info._strides
        ]
        self._computed = np.lib.stride_tricks.as_strided(
            array_with_offset,
            shape=self._tensor_info._shape,
            strides=strides
        ).copy()
        return self._computed

    def to_numpy(self) -> np.ndarray:
        return self._read_from_buffer().copy()

    def __str__(self) -> str:
        return str(self._read_from_buffer())

    def __repr__(self) -> str:
        return repr(self._read_from_buffer())

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tensor_info._shape

    @property
    def dtype(self) -> dt.dtypes.DannetDtype:
        return self._tensor_info._dtype

    @property
    def strides(self) -> tuple[int, ...]:
        return self._tensor_info._strides

    @property
    def ndim(self) -> int:
        return len(self._tensor_info._shape)

    @property
    def size(self) -> int:
        result = 1
        for dim in self._tensor_info._shape:
            result *= dim
        return result

    @property
    def itemsize(self) -> int:
        return dt.dtypes.itemsize(self.dtype)

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    @property
    def device(self) -> dt.Device:
        return self._buffer.device

    def __neg__(self) -> Tensor:
        return dt.negative(self)

    def __add__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.add(self, other)

    def __radd__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.add(other, self)

    def __sub__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.subtract(self, other)

    def __rsub__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.subtract(other, self)

    def __mul__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.multiply(self, other)

    def __rmul__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.multiply(other, self)

    def __truediv__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.divide(self, other)

    def __rtruediv__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.divide(other, self)

    def __pow__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.power(self, other)

    def __rpow__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.power(other, self)

    def __eq__(self, other: dt.typing.TensorLike) -> Tensor:  # type: ignore
        return dt.equal(self, other)
    
    def __neq__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.not_equal(self, other)
    
    def __lt__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.less(self, other)
    
    def __le__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.less_equal(self, other)
    
    def __gt__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.greater(self, other)
    
    def __ge__(self, other: dt.typing.TensorLike) -> Tensor:
        return dt.greater_equal(self, other)
    
    def astype(self, dtype: dt.typing.DTypeLikeO = None) -> Tensor:
        return dt.astype(self, dtype)

    def copy(self) -> Tensor:
        return dt.copy(self)
    
    def reshape(self, shape: dt.typing.ShapeLike) -> Tensor:
        return dt.reshape(self, shape)


def array(
    object: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLike | None = None,
    device: dt.Device | None = None
) -> Tensor:
    if device is None:
        device = dt.current_device()

    with device:
        if isinstance(object, Tensor):
            if (
                (dtype is None or object.dtype == dtype) and
                object.device == device
            ):
                return object
            return object.astype(dtype)

        object = np.asarray(object, dtype=dtype)
        buffer = device.allocate_buffer(object.nbytes)
        dt.device.write_buffer(buffer, object)
        return Tensor(
            buffer, TensorInfo.from_numpy(object), event=None
        )


def empty(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DTypeLike,
    device: dt.Device | None = None
) -> Tensor:
    if device is None:
        device = dt.current_device()
    shape = dt.utils.normalize_shape(shape)
    dtype = dt.utils.normalize_dtype(dtype)

    size = 1
    for dim in shape:
        size *= dim

    buffer = device.allocate_buffer(size * np.dtype(dtype).itemsize)
    return Tensor(
        buffer,
        TensorInfo(shape, dtype),
        event=None
    )


def empty_like(
    x: dt.typing.TensorLike,
    dtype: dt.typing.DTypeLikeO = None,
    device: dt.Device | None = None
) -> Tensor:
    if not isinstance(x, Tensor):
        x = np.array(x)
    shape = x.shape
    dtype = dtype or x.dtype
    return empty(shape, dtype, device)


__all__ = [
    "array",
    "empty", "empty_like"
]
