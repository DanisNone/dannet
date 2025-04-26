import os
import pyopencl as cl

import dannet as dt


class Device:
    _instances = {}
    _stack = []

    def __new__(cls, platform_id=0, device_id=0):
        key = (platform_id, device_id)
        if key not in cls._instances:
            inst = super().__new__(cls)
            cls._instances[key] = inst
        return cls._instances[key]

    def __init__(self, platform_id=0, device_id=0):
        if getattr(self, '_initialized', False):
            return

        self.platform_id = platform_id
        self.device_id = device_id

        platforms = cl.get_platforms()
        if platform_id < 0 or platform_id >= len(platforms):
            raise IndexError(
                f'Platform ID {platform_id} out of range; available: 0..{len(platforms)-1}'
            )
        self.platform = platforms[platform_id]

        devices = self.platform.get_devices()
        if device_id < 0 or device_id >= len(devices):
            raise IndexError(
                f'Device ID {device_id} out of range for platform {platform_id}; available: 0..{len(devices)-1}'
            )
        self.device = devices[device_id]

        self.context = cl.Context(devices=[self.device])
        self.queue = cl.CommandQueue(self.context, self.device)

        self.max_work_group_size = self.device.max_work_group_size
        self._initialized = True

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

        types = [name for flag, name in type_flags.items() if self.device.type & flag]
        type_str = '|'.join(types) if types else str(self.device.type)
        return (
            f'<Device platform={self.platform.name} '
            f'device={self.device.name} '
            f'type={type_str}>'
        )

    def is_support(self, dtype):
        dtype = dt.dtype.normalize_dtype(dtype)
        if dtype == 'float64':
            return 'cl_khr_fp64' in self.device.extensions
        if dtype == 'float16':
            return 'cl_khr_fp16' in self.device.extensions
        return True


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
