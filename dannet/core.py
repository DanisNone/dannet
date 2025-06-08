from __future__ import annotations
import numpy as np

import dannet as dt


class TensorInfo:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: dt.dtypes.DannetDtype,
        strides: tuple[int, ...] | None = None,
        buffer_offset: int = 0
    ):
        if min(shape, default=1) < 0:
            raise ValueError(
                f'All dims of Tensor must be non negative: {shape=}'
            )

        self._shape = shape
        self._dtype = dtype

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
    def get_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
        strides: list[int] = []
        s = 1
        for dim in shape:
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

        itemsize = np.dtype(self._tensor_info._dtype).itemsize
        assert self._buffer.nbytes % itemsize == 0

        array = np.empty(
            self._buffer.nbytes // itemsize,
            dtype=self._tensor_info._dtype
        )
        dt.device.read_buffer(self._buffer, array)

        array_with_offset = array[self._tensor_info._buffer_offset * itemsize:]

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
        result: int = 1
        for dim in self._tensor_info._shape:
            result *= dim
        return result

    @property
    def device(self) -> dt.Device:
        return self._buffer.device


def array(
    object: dt.typing.TensorLike,
    dtype: dt.typing.DtypeLike | None = None,
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

        if device is None:
            device = dt.current_device()
        object = np.array(object)
        buffer = device.allocate_buffer(object.nbytes)
        dt.device.write_buffer(buffer, object)
        return Tensor(
            buffer, TensorInfo.from_numpy(object), event=None
        )


def empty(
    shape: dt.typing.ShapeLike,
    dtype: dt.typing.DtypeLike,
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
    dtype: dt.typing.DtypeLikeO = None,
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
