from __future__ import annotations
import types
from typing import Any
from weakref import WeakSet

import pyopencl as cl
import numpy as np


class BuildError(Exception):
    pass


class Device:
    __instances: dict[
        tuple[int, int], Device
    ] = {}
    __stack: list[Device] = []

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

    def __eq__(self, other: Any) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __device_init__(self, platform_id: int, device_id: int) -> None:
        platforms = cl.get_platforms()
        if not (0 <= platform_id < len(platforms)):
            raise ValueError(
                f"platform count: {len(platforms)}. "
                f"given {platform_id=}"
            )
        self.__cl_platform = platforms[platform_id]

        devices = self.__cl_platform.get_devices()
        if not (0 <= device_id < len(devices)):
            raise ValueError(
                f"device count: {len(devices)}. "
                f"given {device_id=}"
            )
        self.__cl_device = devices[device_id]

        self.__cl_context = cl.Context([self.__cl_device])
        self.__cl_queue = cl.CommandQueue(self.__cl_context, self.__cl_device)

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

    def __get_cl_platform__(self) -> cl.Platform:
        return self.__cl_platform

    def __get_cl_device__(self) -> cl.Device:
        return self.__cl_device

    def __get_cl_context__(self) -> cl.Context:
        return self.__cl_context

    def __get_cl_queue__(self) -> cl.CommandQueue:
        return self.__cl_queue

    def __enter__(self) -> Device:
        self.__class__.__stack.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None
    ) -> None:
        popped = self.__class__.__stack.pop()
        if popped is not self:
            self.__class__.__stack.append(popped)
            raise RuntimeError('Device stack corrupted on exit')

    @classmethod
    def current_device(cls) -> Device:
        return cls.__stack[-1] if cls.__stack else default_device()

    def __repr__(self) -> str:
        type_flags = {
            cl.device_type.CPU: 'CPU',
            cl.device_type.GPU: 'GPU',
            cl.device_type.ACCELERATOR: 'ACCELERATOR',
            cl.device_type.DEFAULT: 'DEFAULT',
            cl.device_type.CUSTOM: 'CUSTOM',
        }

        types = [name for flag, name in type_flags.items()
                 if self.__cl_device.type & flag]
        type_str = '|'.join(types) if types else str(self.__cl_device.type)
        return (
            f'<Device platform={self.__cl_platform.name} '
            f'device={self.__cl_device.name} '
            f'type={type_str}>'
        )


class DeviceBuffer:
    def __init__(
        self,
        device: Device,
        nbytes: int
    ):
        self.__device = device
        self.__nbytes = nbytes
        self.__released = False

        if nbytes < 0:
            raise ValueError(f"nbytes must be non negative: {nbytes=}")

        if nbytes == 0:
            self.__cl_buffer = None
        else:
            self.__cl_buffer = cl.Buffer(
                device.__get_cl_context__(),
                cl.mem_flags.READ_WRITE,
                nbytes
            )

    def __get_cl_buffer__(self) -> cl.Buffer | None:
        return self.__cl_buffer

    @property
    def device(self) -> Device:
        return self.__device

    @property
    def nbytes(self) -> int:
        return self.__nbytes

    @property
    def released(self) -> bool:
        return self.__released

    def release(self) -> None:
        if not self.__released:
            if self.__cl_buffer is not None:
                self.__cl_buffer.release()
            self.__device._release_buffer(self)
            self.__released = True

    def __del__(self) -> None:
        self.release()


class DeviceEvent:
    def __init__(self, events: list[cl.Event | DeviceEvent]):
        self.__events: list[cl.Event] = []
        for event in events:
            if isinstance(event, DeviceEvent):
                self.__events.extend(event.__events)
            else:
                self.__events.append(event)

    def __get_cl_events__(self) -> list[cl.Event]:
        return self.__events

    def wait(self) -> None:
        for event in self.__events:
            event.wait()


def empty_event() -> DeviceEvent:
    return DeviceEvent([])


class DeviceProgram:
    def __init__(self, device: Device, source: str):
        self.__device = device
        self.__cl_program: cl.Program = cl.Program(
            self.__device.__get_cl_context__(),
            source
        )

    def build(self, options: list[str] = []) -> None:
        try:
            self.__cl_program.build(options)
        except Exception as e:
            raise BuildError(f"Program build failed: {e}")

    def __get_cl_kernel__(self, name: str) -> cl.Kernel:
        try:
            kernel: cl.Kernel = getattr(self.__cl_program, name)
        except AttributeError:
            raise AttributeError(
                f"'DeviceProgram' object has no attribute '{name}'"
            )
        return kernel

    def __getattr__(self, attr: str) -> DeviceKernel:
        return DeviceKernel(self, attr)

    @property
    def device(self) -> Device:
        return self.__device


class DeviceKernel:
    def __init__(
        self,
        program: DeviceProgram,
        name: str
    ):
        self.__device = program.device
        self.__program = program
        self.__name = name
        self.__cl_kernel = program.__get_cl_kernel__(name)

    def __call__(
        self,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
        wait_for: DeviceEvent | None = None
    ) -> DeviceEvent:
        wait_event = None if wait_for is None else wait_for.__events

        new_args: list[Any] = []
        for arg in args:
            if isinstance(arg, DeviceBuffer):
                if arg.device != self.__device:
                    raise ValueError("")
                arg = arg.__cl_buffer
            new_args.append(args)

        event = self.__cl_kernel.__call__(
            self.__device.__get_cl_queue__(),
            global_size, local_size,
            *new_args,
            wait_for=wait_event
        )
        return DeviceEvent([event])


def read_buffer(buffer: DeviceBuffer, array: np.ndarray) -> None:
    if buffer.nbytes != array.nbytes:
        raise ValueError(
            "buffer and array nbytes not equal: "
            f"{buffer.nbytes=}; "
            f"{array.nbytes=}"
        )

    cl_buffer = buffer.__get_cl_buffer__()
    if cl_buffer is None:
        return

    cl.enqueue_copy(
        buffer.device.__get_cl_queue__(),
        array, cl_buffer,
        is_blocking=True
    )


def write_buffer(buffer: DeviceBuffer, array: np.ndarray) -> None:
    if buffer.nbytes != array.nbytes:
        raise ValueError(
            "buffer and array nbytes not equal: "
            f"{buffer.nbytes=}; "
            f"{array.nbytes=}"
        )

    cl_buffer = buffer.__get_cl_buffer__()
    if cl_buffer is None:
        return

    cl.enqueue_copy(
        buffer.device.__get_cl_queue__(),
        cl_buffer, array,
        is_blocking=True
    )


def default_device() -> Device:
    # TODO: add os.environ support
    return Device(0, 0)


current_device = Device.current_device
