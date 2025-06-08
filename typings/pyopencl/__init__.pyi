from __future__ import annotations
from typing import Any, NoReturn, overload

import numpy as np


class Platform:
    name: str
    def get_devices(self) -> list[Device]: ...


class Device:
    type: int
    name: str
    address_bits: int


class Context:
    def __init__(
        self: Context,
        devices: list[Device] | None = None,
        properties: list[tuple[int, Any]] | None = None,
        dev_type: int | None = None
    ) -> None: ...


class CommandQueue:
    def __init__(
        self,
        context: Context,
        device: Device | None = None,
        properties: int | None = None
    ): ...


class Buffer:
    def __init__(
        self,
        context: Context,
        flags: int,
        size: int = 0,
        hostbuf: np.ndarray | None = None
    ): ...
    def release(self) -> None: ...


class Event:
    def __init__(self) -> NoReturn: ...
    def wait(self) -> None: ...


class Program:
    @overload
    def __init__(
        self,
        context: Context,
        src: bytes | str
    ): ...

    @overload
    def __init__(
        self,
        devices: list[Device],
        binaries: bytes
    ): ...

    def __getattr__(self, attr: str) -> Kernel: ...

    def build(
        self,
        options: list[str] = [],
        devices: list[Device] | None = None,
        cache_dir: str | None = None
    ) -> None: ...


class Kernel:
    def __init__(
        self,
        program: Program,
        name: str
    ): ...

    def __call__(
        self,
        queue: CommandQueue,
        global_size: tuple[int, ...],
        local_size: tuple[int, ...] | None,
        *args: Any,
        wait_for: list[Event] | None = None
    ) -> Event: ...


def get_platforms() -> list[Platform]: ...


@overload
def enqueue_copy(
    queue: CommandQueue,
    dest: np.ndarray,
    src: Buffer,
    wait_for: list[Event] = [],
    is_blocking: bool = True
) -> Event: ...


@overload
def enqueue_copy(
    queue: CommandQueue,
    dest: Buffer,
    src: np.ndarray,
    wait_for: list[Event] = [],
    is_blocking: bool = True
) -> Event: ...


class mem_flags:
    READ_WRITE: int

class device_type:
    CPU: int
    GPU: int
    ACCELERATOR: int
    DEFAULT: int
    CUSTOM: int