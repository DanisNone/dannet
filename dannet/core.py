from __future__ import annotations
import numpy as np

import dannet as dt


class TensorInfo:
    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: dt.dtypes.DannetDtype,
        strides: tuple[int, ...],
        buffer_offset: int = 0
    ):
        if min(shape, default=1) < 0:
            raise ValueError(
                f'All dims of Tensor must be non negative: {shape=}'
            )

        self._shape = shape
        self._dtype = dtype
        self._strides = strides

        if buffer_offset < 0:
            raise ValueError(
                f'buffer offset must be non negative: {buffer_offset=}'
            )
        self._buffer_offset = buffer_offset

    @classmethod
    def from_numpy(cls, array: np.ndarray) -> TensorInfo:
        strides: list[int] = []
        s = 1
        for dim in array.shape:
            strides.append(s)
            s *= dim
        strides.reverse()

        return cls(
            shape=array.shape,
            dtype=array.dtype,
            strides=tuple(strides),
            buffer_offset=0
        )


class Tensor:
    def __init__(
        self,
        buffer: dt.device.DeviceBuffer,
        tensor_info: TensorInfo,
        event: dt.device.DeviceEvent | None,
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
