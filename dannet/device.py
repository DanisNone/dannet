from __future__ import annotations
from typing import Any
from weakref import WeakSet

import pyopencl as cl
import numpy as np


class Device:
    __instances: dict[
        tuple[int, int], Device
    ] = {}

    def __new__(cls, platform_id: int, device_id: int) -> Device:
        platform_id = int(platform_id)
        device_id = int(device_id)

        if platform_id < 0:
            raise ValueError(
                f"platform_id must be non negative: {platform_id=}"
            )
        if device_id < 0:
            raise ValueError(f"device_id must be non negative: {device_id=}")

        if (platform_id, device_id) not in cls.__instances:
            instance = super().__new__(cls)
            instance.__device_init__(platform_id, device_id)
            cls.__instances[(platform_id, device_id)] = instance
        return cls.__instances[(platform_id, device_id)]

    def __device_init__(self, platform_id: int, device_id: int) -> None:
        platforms = cl.get_platforms()
        if not (0 <= platform_id < len(platforms)):
            raise ValueError(
                f"platform count: {len(platforms)}. "
                f"given {platform_id=}"
            )
        self.__platform = platforms[platform_id]

        devices = self.__platform.get_devices()
        if not (0 <= device_id < len(devices)):
            raise ValueError(
                f"device count: {len(devices)}. "
                f"given {device_id=}"
            )
        self.__device = devices[device_id]

        self.__context = cl.Context([self.__device])
        self.__queue = cl.CommandQueue(self.__context, self.__device)

        self.__allocated_buffers: WeakSet[DeviceBuffer] = WeakSet()
        self.__allocated_memory: int = 0

    def allocate_buffer(self, nbytes: int) -> DeviceBuffer:
        if nbytes < 0:
            raise ValueError(f"nbytes must be non negative: {nbytes=}")

        buffer = DeviceBuffer(self, nbytes)
        self.__allocated_buffers.add(buffer)
        self.__allocated_memory += nbytes
        return buffer

    def _release_buffer(self, buffer: DeviceBuffer) -> None:
        if not buffer.released:
            self.__allocated_memory -= buffer.nbytes

    @property
    def platform(self) -> cl.Platform:
        return self.__platform

    @property
    def device(self) -> cl.Device:
        return self.__device

    @property
    def context(self) -> cl.Context:
        return self.__context

    @property
    def queue(self) -> cl.CommandQueue:
        return self.__queue


class DeviceBuffer:
    def __init__(
        self,
        device: Device,
        nbytes: int
    ):
        self._device = device
        self._nbytes = nbytes
        self._released = False

        if nbytes < 0:
            raise ValueError(f"nbytes must be non negative: {nbytes=}")

        if nbytes == 0:
            self.cl_buffer = None
        else:
            self.cl_buffer = cl.Buffer(
                device.context,
                cl.mem_flags.READ_WRITE,
                nbytes
            )

    @property
    def device(self) -> Device:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        if not self._released:
            if self.cl_buffer is not None:
                self.cl_buffer.release()
            self._device._release_buffer(self)
            self._released = True

    def __del__(self) -> None:
        self.release()


class DeviceEvent:
    def __init__(self, events: list[cl.Event | DeviceEvent]):
        self.events: list[cl.Event] = []
        for event in events:
            if isinstance(event, DeviceEvent):
                self.events.extend(event.events)
            else:
                self.events.append(event)

    def wait(self) -> None:
        for event in self.events:
            event.wait()


def empty_event() -> DeviceEvent:
    return DeviceEvent([])


class DeviceKernel:
    def __init__(self, kernel: cl.Kernel):
        self.kernel = kernel

    def __call__(
        self,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...],
        *args: Any
    ) -> DeviceEvent:
        raise NotImplementedError


def read_buffer(buffer: DeviceBuffer, array: np.ndarray) -> None:
    if buffer.nbytes != array.nbytes:
        raise ValueError(
            "buffer and array nbytes not equal: "
            f"{buffer.nbytes=}; "
            f"{array.nbytes=}"
        )

    if buffer.cl_buffer is None:
        return

    cl.enqueue_copy(
        buffer.device.queue,
        array, buffer.cl_buffer,
        is_blocking=True
    )


def write_buffer(buffer: DeviceBuffer, array: np.ndarray) -> None:
    if buffer.nbytes != array.nbytes:
        raise ValueError(
            "buffer and array nbytes not equal: "
            f"{buffer.nbytes=}; "
            f"{array.nbytes=}"
        )
    if buffer.cl_buffer is None:
        return

    cl.enqueue_copy(
        buffer.device.queue,
        buffer.cl_buffer, array,
        is_blocking=True
    )


def default_device() -> Device:
    # TODO: add os.environ support
    return Device(0, 0)
