# TODO: Implement DeviceEvent class
from __future__ import annotations

import enum
import os
from typing import overload
from weakref import WeakSet
import numpy as np
import pyopencl as cl

import dannet as dt


class mem_flags(enum.IntFlag):
    READ_ONLY = cl.mem_flags.READ_ONLY
    WRITE_ONLY = cl.mem_flags.WRITE_ONLY
    READ_WRITE = cl.mem_flags.READ_WRITE


class Device:
    _instances: dict[tuple[int, int], Device] = {}
    _stack: list[Device] = []

    def __new__(cls, platform_id=0, device_id=0):
        key = (platform_id, device_id)
        if key not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, platform_id=0, device_id=0) -> None:
        if getattr(self, '_initialized', False):
            return

        self.platform_id: int = platform_id
        self.device_id: int = device_id

        platforms = cl.get_platforms()
        if platform_id < 0 or platform_id >= len(platforms):
            raise IndexError(
                f'Platform ID {platform_id} out of range; '
                f'available: 0..{len(platforms)-1}'
            )
        self.platform: cl.Platform = platforms[platform_id]

        devices = self.platform.get_devices()
        if device_id < 0 or device_id >= len(devices):
            raise IndexError(
                f'Device ID {device_id} out of range '
                f'for platform {platform_id}; '
                f'available: 0..{len(devices)-1}'
            )
        self.device: cl.Device = devices[device_id]

        self.context: cl.Context = cl.Context(devices=[self.device])
        self.queue: cl.CommandQueue = cl.CommandQueue(
            self.context, self.device
        )

        self.max_work_group_size: int = self.device.max_work_group_size
        self.allocated_buffers: WeakSet[DeviceBuffer] = WeakSet()

        self.memory_usage: int = 0
        self._initialized: bool = True

    def __enter__(self):
        self.__class__._stack.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        popped = self.__class__._stack.pop()
        if popped is not self:
            self.__class__._stack.append(popped)
            raise RuntimeError('Device stack corrupted on exit')

    @classmethod
    def current_device(cls):
        return cls._stack[-1] if cls._stack else default_device()

    def __repr__(self):
        type_flags = {
            cl.device_type.CPU: 'CPU',
            cl.device_type.GPU: 'GPU',
            cl.device_type.ACCELERATOR: 'ACCELERATOR',
            cl.device_type.DEFAULT: 'DEFAULT',
            cl.device_type.CUSTOM: 'CUSTOM',
        }

        types = [name for flag, name in type_flags.items()
                 if self.device.type & flag]
        type_str = '|'.join(types) if types else str(self.device.type)
        return (
            f'<Device platform={self.platform.name} '
            f'device={self.device.name} '
            f'type={type_str}>'
        )

    def is_support(self, dtype: dt.typing.DTypeLike) -> bool:
        dtype = dt.dtype.normalize_dtype(dtype)
        if dtype == dt.dtype.float64:
            return 'cl_khr_fp64' in self.device.extensions
        if dtype == dt.dtype.float16:
            return 'cl_khr_fp16' in self.device.extensions
        return True

    @staticmethod
    def _format_bytes(size: float | int) -> str:
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        while size >= 1000 and unit_index < len(units)-1:
            unit_index += 1
            size /= 1000.0
        return f'{size:.2f} {units[unit_index]}'

    def _format_memory_error(self, memory_usage, nbytes):
        format_bytes = self._format_bytes
        return (
            f'Not enough device memory:\n'
            f'  Current usage: {format_bytes(memory_usage)}\n'
            f'  Requested allocation: {format_bytes(nbytes)}\n'
            f'  Total after allocation: '
            f'{format_bytes(memory_usage + nbytes)}\n'
            f'  Device limit: {format_bytes(self.device.global_mem_size)}'
        )

    def allocate_buffer(self, flag: mem_flags, nbytes: int) -> DeviceBuffer:
        nbytes = int(nbytes)
        if self.memory_usage + nbytes >= self.device.global_mem_size:
            msg = self._format_memory_error(self.memory_usage, nbytes)
            raise MemoryError(msg)
        buffer = DeviceBuffer(self, flag, nbytes)
        self.memory_usage += nbytes
        self.allocated_buffers.add(buffer)

        return buffer

    def free_buffer(self, buffer: DeviceBuffer):
        if not isinstance(buffer, DeviceBuffer):
            raise TypeError(
                f'Argument must be DeviceBuffer, got {type(buffer).__name__}'
            )

        if buffer.released:
            raise RuntimeError(
                f'Buffer {buffer} has already been released'
            )

        if buffer not in self.allocated_buffers:
            raise ValueError(
                f'Buffer {buffer} is not allocated by this device '
                f'(platform={self.platform_id}, device={self.device_id})'
            )

        self.memory_usage -= buffer.nbytes
        self.allocated_buffers.remove(buffer)
        buffer._released = True

    @overload
    def enqueue_copy(
        self,
        dest: DeviceBuffer,
        src: np.ndarray | dt.core.Constant,
        *, is_blocking: bool = True
    ) -> cl.Event: ...

    @overload
    def enqueue_copy(
        self,
        dest: np.ndarray,
        src: DeviceBuffer,
        *, is_blocking: bool = True
    ) -> cl.Event: ...

    def enqueue_copy(self, dest, src, *, is_blocking=True) -> cl.Event:
        dest_is_buf = isinstance(dest, DeviceBuffer)
        dest_is_host = isinstance(dest, np.ndarray)

        src_is_buf = isinstance(src, DeviceBuffer)
        src_is_host = isinstance(src, (np.ndarray, dt.core.Constant))

        if not (dest_is_buf or dest_is_host):
            raise TypeError(f'Invalid destination type: {type(dest).__name__}')
        if not (src_is_buf or src_is_host):
            raise TypeError(f'Invalid source type: {type(src).__name__}')

        if not (
            (dest_is_buf and src_is_host) or
            (dest_is_host and src_is_buf)
        ):
            raise TypeError(
                f'Invalid copy direction. Expected either:\n'
                f'- (DeviceBuffer, host_array) or\n'
                f'- (host_array, DeviceBuffer)\n'
                f'Got: dest={type(dest).__name__}, src={type(src).__name__}'
            )

        if src_is_buf:
            src_norm = src.cl_buffer
        else:
            src_norm = src._value if isinstance(
                src, dt.core.Constant) else src.copy()

        if dest_is_buf:
            dest_norm = dest.cl_buffer
        else:
            dest_norm = dest

        if dest_is_buf and (dest.nbytes != src.nbytes):
            raise ValueError(
                f'Buffer size mismatch. DeviceBuffer: {dest.nbytes} bytes, '
                f'Host array: {src.nbytes} bytes'
            )

        if src_is_buf and (dest.nbytes != src.nbytes):
            raise ValueError(
                f'Buffer size mismatch. Host array: {dest.nbytes} bytes, '
                f'DeviceBuffer: {src.nbytes} bytes'
            )

        return cl.enqueue_copy(
            queue=self.queue,
            dest=dest_norm,
            src=src_norm,
            is_blocking=is_blocking
        )


class DeviceBuffer:
    def __init__(self, device: Device, flags: mem_flags, nbytes: int):
        self._device = device
        self._cl_buffer = cl.Buffer(device.context, int(flags), nbytes)
        self._released = False

    @property
    def nbytes(self) -> int:
        return self._cl_buffer.size

    @property
    def cl_buffer(self) -> cl.Buffer:
        return self._cl_buffer

    @property
    def released(self) -> bool:
        return self._released

    def release(self):
        if not self._released:
            self._device.free_buffer(self)

    def __del__(self):
        # automatically free on garbage collection
        try:
            self.release()
        except Exception:
            pass


def default_device() -> Device:
    try:
        platform_id = int(os.getenv('DANNET_DEFAULT_PLATFORM_ID', '0'))
    except ValueError:
        platform_id = 0
    try:
        device_id = int(os.getenv('DANNET_DEFAULT_DEVICE_ID', '0'))
    except ValueError:
        device_id = 0
    return Device(platform_id, device_id)


current_device = Device.current_device
