import numpy as np
import functools
from pathlib import Path
import dannet as dt
import ctypes

from dannet.lib import dtypes
import dannet.lib


class ShapeInfo(ctypes.Structure):
    pass


class ShapeInfoX64(ShapeInfo):
    _fields_ = [
        ("buffer_offset", ctypes.c_uint64),
        ("ndim", ctypes.c_uint64),
        ("shape", ctypes.c_uint64 * 64),
        ("strides", ctypes.c_uint64 * 64),
    ]


class ShapeInfoX32(ShapeInfo):
    _fields_ = [
        ("buffer_offset", ctypes.c_uint32),
        ("ndim", ctypes.c_uint32),
        ("shape", ctypes.c_uint32 * 64),
        ("strides", ctypes.c_uint32 * 64),
    ]


class Shape(ctypes.Structure):
    pass


class ShapeX64(Shape):
    _fields_ = [
        ("ndim", ctypes.c_uint64),
        ("data", ctypes.c_uint64 * 64),
    ]


class ShapeX32(Shape):
    _fields_ = [
        ("ndim", ctypes.c_uint32),
        ("data", ctypes.c_uint32 * 64),
    ]


def get_shape_info(device: dt.Device, x: dannet.lib.core.ConcreteTensor) -> ShapeInfo:
    address_bits = device.__get_cl_device__().address_bits
    info: ShapeInfo
    if address_bits == 64:
        info = ShapeInfoX64()
    elif address_bits == 32:
        info = ShapeInfoX32()
    else:
        raise ValueError(
            f"not support address bits of device: {address_bits=}"
        )

    info.buffer_offset = x.buffer_offset

    ndim = x.ndim
    if ndim > 64:
        raise ValueError(
            "maximum supported dimension for an Tensor "
            f"is currently 64. {ndim=}"
        )
    info.ndim = ndim

    shape_tuple = x.shape
    strides_tuple = x.strides

    for i in range(ndim):
        info.shape[i] = shape_tuple[i]
        info.strides[i] = strides_tuple[i]

    return info


def get_shape(device: dt.Device, x: tuple[int, ...]) -> Shape:
    address_bits = device.__get_cl_device__().address_bits
    shape: Shape
    if address_bits == 64:
        shape = ShapeX64()
    elif address_bits == 32:
        shape = ShapeX32()
    else:
        raise ValueError(
            f"not support address bits of device: {address_bits=}"
        )

    ndim = len(x)
    if ndim > 64:
        raise ValueError(
            "maximum supported dimension for an Tensor "
            f"is currently 64. {ndim=}"
        )

    shape.ndim = ndim
    for i in range(ndim):
        shape.data[i] = x[i]
    return shape


def get_size_t(device: dt.Device, x: int) -> np.uint32 | np.uint64:
    dtype = np.uint64 if device.__get_cl_device__().address_bits == 64 else np.uint32
    return dtype(x)


class BuildInfo:
    def __init__(self) -> None:
        self.dtypes: dict[str, dtypes.DannetDtype] = {}
        self.headers: list[str] = []

    def add_dtypes(self, **kwargs: dtypes.DannetDtype) -> None:
        for key, value in kwargs.items():
            self.dtypes[key] = value

    def add_header(self, s: str) -> None:
        self.headers.append(s)

    def build(self, source: str, device: dt.Device) -> str:
        _dtypes = self.dtypes
        _dtypes["size_t"] = dtypes.uint64 if device.__get_cl_device__(
        ).address_bits == 64 else dtypes.uint32

        headers = ['#include "dtypes/core.cl"'] + self.headers
        source = "\n".join(headers + [source])
        for dtype_name, dtype in _dtypes.items():
            source = source.replace(
                f"${dtype_name}$",
                dtype.__name__
            )

        return source


@functools.cache
def _build_from_source(
    device: dt.Device, source: str, options: tuple[str, ...] = ()
) -> dt.device.DeviceProgram:
    program = dt.device.DeviceProgram(device, source)
    program.build(list(options))
    return program


def build_program(
    device: dt.Device,
    path: str,
    build_info: BuildInfo
) -> dt.device.DeviceProgram:

    root = Path(__file__).parent.parent
    with open(root / "kernels" / path) as file:
        source = file.read()

    source = build_info.build(source, device)

    with open("last_compiled.cl", "w") as file:
        file.write(source)
    options = (
        f"-I {root}",
    )
    return _build_from_source(device, source, options)
