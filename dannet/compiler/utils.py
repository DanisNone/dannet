import functools
from pathlib import Path
import dannet as dt
import ctypes


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


def get_shape_info(device: dt.Device, x: dt.core.Tensor) -> ShapeInfo:
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

    info.buffer_offset = x._tensor_info._buffer_offset

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


class BuildInfo:
    def __init__(self) -> None:
        self.dtypes: dict[str, dt.dtypes.DannetDtype] = {}
        self.headers: list[str] = []

    def add_dtypes(self, **kwargs: dt.dtypes.DannetDtype) -> None:
        for key, value in kwargs.items():
            self.dtypes[key] = value

    def add_header(self, s: str) -> None:
        self.headers.append(s)

    def build(self, source: str) -> str:
        source = "\n".join(self.headers + [source])
        for dtype_name, dtype in self.dtypes.items():
            source = source.replace(
                f"${dtype_name}$",
                dtype.__name__
            )
        return source


@functools.lru_cache
def _build_from_source(
    device: dt.Device, source: str
) -> dt.device.DeviceProgram:
    program = dt.device.DeviceProgram(device, source)
    program.build()
    return program


def build_program(
    device: dt.Device,
    path: str,
    build_info: BuildInfo
) -> dt.device.DeviceProgram:

    root = Path(__file__).parent / "kernels"
    with open(root / path) as file:
        source = file.read()

    source = build_info.build(source)
    return _build_from_source(device, source)
