from __future__ import annotations

import abc
from types import EllipsisType
from typing import Any, Hashable, Sequence, SupportsIndex, TypeAlias
import numpy as np

import dannet as dt
import dannet.device
from dannet.lib import dtypes
from dannet.lib.dtypes import DannetDtype
from dannet.lib.dtypes import DTypeLike
from dannet.device import Device, DeviceEvent
from dannet.device import DeviceBuffer

Axes: TypeAlias = Sequence[SupportsIndex] | SupportsIndex | None


class BaseBuffer(abc.ABC):
    @property
    @abc.abstractmethod
    def nbytes(self) -> int: ...


class SymbolicBuffer(BaseBuffer):
    def __init__(
        self,
        tensor: SymbolicTensor,
        nbytes: int | None = None
    ):
        if nbytes is None:
            nbytes = tensor.nbytes
        self._tensor = tensor
        self._nbytes = nbytes

        if nbytes < 0:
            raise ValueError(f"nbytes must be non negative: {nbytes=}")

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __graph_eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        return self.nbytes == other.nbytes and self._tensor.__graph_eq__(other._tensor)

    def __graph_hash__(self) -> int:
        return hash((self._tensor.__graph_hash__(), self.nbytes))


class ConcreteBuffer(DeviceBuffer, BaseBuffer):
    pass


_getitem_sub_type = (
    int | slice | None |
    EllipsisType
)
_getitem_key_type = (
    _getitem_sub_type |
    tuple[_getitem_sub_type, ...]
)


class BaseTensor(abc.ABC):
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _offset: int
    _dtype: DannetDtype

    @abc.abstractmethod
    def _maybe_concrete(self) -> bool: ...

    @property
    @abc.abstractmethod
    def buffer(self) -> BaseBuffer: ...

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def buffer_offset(self) -> int:
        return self._offset

    @property
    def dtype(self) -> DannetDtype:
        return self._dtype

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        size = 1
        for dim in self._shape:
            size *= dim
        return size

    @property
    def itemsize(self) -> int:
        return dtypes.itemsize(self._dtype)

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def __bool__(self) -> bool:
        if self.ndim != 0:
            raise ValueError(
                "The truth value of an array "
                "with more than one element is ambiguous. "
                "Use a.any() or a.all()"
            )
        if not isinstance(self, ConcreteTensor):
            if self._maybe_concrete():
                raise NotImplementedError
            raise ValueError
        return bool(self.__array__())

    def __eq__(self, other: TensorLike) -> BaseTensor:  # type: ignore
        return dannet.equal(self, other)

    def __ne__(self, other: TensorLike) -> BaseTensor:  # type: ignore
        return dannet.not_equal(self, other)

    def __lt__(self, other: TensorLike) -> BaseTensor:
        return dannet.less(self, other)

    def __le__(self, other: TensorLike) -> BaseTensor:
        return dannet.less_equal(self, other)

    def __gt__(self, other: TensorLike) -> BaseTensor:
        return dannet.greater(self, other)

    def __ge__(self, other: TensorLike) -> BaseTensor:
        return dannet.greater_equal(self, other)

    def __add__(self, other: TensorLike) -> BaseTensor:
        return dannet.add(self, other)

    def __radd__(self, other: TensorLike) -> BaseTensor:
        return dannet.add(other, self)

    def __sub__(self, other: TensorLike) -> BaseTensor:
        return dannet.subtract(self, other)

    def __rsub__(self, other: TensorLike) -> BaseTensor:
        return dannet.subtract(other, self)

    def __mul__(self, other: TensorLike) -> BaseTensor:
        return dannet.multiply(self, other)

    def __rmul__(self, other: TensorLike) -> BaseTensor:
        return dannet.multiply(other, self)

    def __truediv__(self, other: TensorLike) -> BaseTensor:
        return dannet.divide(self, other)

    def __rtruediv__(self, other: TensorLike) -> BaseTensor:
        return dannet.divide(other, self)

    def __matmul__(self, other: TensorLike) -> BaseTensor:
        return dannet.matmul(self, other)

    def __rmatmul__(self, other: TensorLike) -> BaseTensor:
        return dannet.matmul(other, self)

    def __neg__(self) -> BaseTensor:
        return dannet.negative(self)

    def __getitem__(
        self,
        key: _getitem_key_type
    ) -> BaseTensor:
        slices, newaxes_positions, squeeze_axes = parse_getitem_key(self, key)
        result = dt.slice(self, slices)

        for axis in newaxes_positions:
            result = dt.expand_dims(result, axis)

        if squeeze_axes:
            result = dt.squeeze(result, squeeze_axes)

        return result

    def astype(self, dtype: DTypeLike | None) -> BaseTensor:
        if dtype is None:
            return self
        return dt.astype(self, dtype)

    def copy(self) -> BaseTensor:
        return dt.copy(self)

    def real(self) -> BaseTensor:
        return dt.real(self)

    def imag(self) -> BaseTensor:
        return dt.imag(self)

    def conj(self) -> BaseTensor:
        return dt.conj(self)

    def conjugate(self) -> BaseTensor:
        return dt.conjugate(self)

    @property
    def at(self) -> dannet.lib.at.AtObject:
        return dannet.lib.at.AtObject(self)


class _Meta(abc.ABCMeta):
    def __call__(cls, *args: Any, **kwargs: Any) -> SymbolicTensor:
        instance: SymbolicTensor = super().__call__(*args, **kwargs)
        if instance.inputs() and instance._maybe_concrete():
            from dannet.compiler import compile
            result = compile(dannet.device.current_device(),
                             [], [instance])([])[0]
            return Constant(result)
        instance.__graph_hash__()
        return instance


class SymbolicTensor(BaseTensor, metaclass=_Meta):
    _buffer: SymbolicBuffer
    _hash: int | None

    @property
    def buffer(self) -> SymbolicBuffer:
        return self._buffer

    def _maybe_concrete(self) -> bool:
        return all(inp._maybe_concrete() for inp in self.inputs())

    @abc.abstractmethod
    def inputs(self) -> list[SymbolicTensor]: ...

    @abc.abstractmethod
    def get_config(self) -> dict[str, Hashable]: ...

    def __graph_eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if type(self) is not type(other):
            return False
        if (
            self.shape != other.shape or
            self.strides != other.strides or
            self.buffer_offset != other.buffer_offset or
            self.dtype != other.dtype or
            self.buffer != other.buffer
        ):
            return False
        if self.get_config() != other.get_config():
            return False
        inputs1, inputs2 = self.inputs(), other.inputs()
        if len(inputs1) != len(inputs2):
            return False
        return all(inp1.__graph_eq__(inp2) for inp1, inp2 in zip(inputs1, inputs2))

    def __graph_hash__(self) -> int:
        if not hasattr(self, "_hash") or self._hash is None:
            self._hash = hash((
                type(self),
                self.shape,
                self.strides,
                self.buffer_offset,
                self.dtype,
                *self.get_config().values(),
                *(inp.__graph_hash__() for inp in self.inputs())
            ))
        return self._hash


class ConcreteTensor(BaseTensor):
    _buffer: ConcreteBuffer

    def __init__(
        self,
        shape: tuple[int, ...],
        strides: tuple[int, ...],
        offset: int,
        dtype: DannetDtype,
        buffer: ConcreteBuffer, /,
        event: DeviceEvent | None
    ):
        self._shape = shape
        self._strides = strides
        self._offset = offset
        self._dtype = dtype
        self._buffer = buffer
        self._event = event

    @classmethod
    def from_ndarray(cls, array: np.ndarray) -> ConcreteTensor:
        buffer = ConcreteBuffer(dannet.device.current_device(), array.nbytes)
        dannet.device.write_buffer(buffer, array)

        return cls(
            array.shape,
            default_strides(array.shape),
            0,
            dtypes.normalize_dtype(array.dtype),
            buffer,
            event=None
        )

    @property
    def buffer(self) -> ConcreteBuffer:
        return self._buffer

    def __array__(self, copy: bool = True) -> np.ndarray:
        if not copy:
            raise ValueError("BaseTensor not support copy=False")
        array = np.empty(self.buffer.nbytes // self.itemsize, dtype=self.dtype)
        dannet.device.read_buffer(self.buffer, array)

        strides = [s * self.itemsize for s in self.strides]
        return np.lib.stride_tricks.as_strided(
            array[self.buffer_offset:],
            self.shape, strides
        ).copy()

    def _maybe_concrete(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self.__array__())

    def __repr__(self) -> str:
        return repr(self.__array__())


class Constant(SymbolicTensor):
    def __init__(self, tensor: ConcreteTensor):
        self._shape = tensor._shape
        self._strides = tensor._strides
        self._offset = tensor._offset
        self._dtype = tensor._dtype
        self._buffer = SymbolicBuffer(self, tensor.buffer.nbytes)

        self._concrete_tensor = tensor

    def inputs(self) -> list[SymbolicTensor]:
        return []

    def get_config(self) -> dict[str, Hashable]:
        return {"id": id(self)}


class Placeholder(SymbolicTensor):
    def __init__(self, x: ConcreteTensor):
        self._shape = x.shape
        self._strides = x.strides
        self._offset = x.buffer_offset
        self._dtype = x.dtype

        self._buffer = SymbolicBuffer(self, x.buffer.nbytes)

    def _maybe_concrete(self) -> bool:
        return False

    def inputs(self) -> list[SymbolicTensor]:
        return []

    def get_config(self) -> dict[str, Hashable]:
        return {"id": id(self)}


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()

    max_ndim = max(len(shape) for shape in shapes)
    padded_shapes = [(1,) * (max_ndim - len(shape)) +
                     shape for shape in shapes]

    result = []
    for dims in zip(*padded_shapes):
        non_one_dims = [d for d in dims if d != 1]
        if not non_one_dims:
            result.append(1)
            continue
        max_dim = non_one_dims[0]
        if any(d != max_dim and d != 1 for d in dims):
            raise ValueError(f"Shapes {shapes} are not broadcast-compatible")
        result.append(max_dim)

    return tuple(result)


def default_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides: list[int] = []
    s = 1
    for dim in shape[::-1]:
        strides.append(s)
        s *= dim
    return tuple(strides[::-1])


def require_equal_shape(name: str, *tensors: BaseTensor) -> None:
    shapes = [tensor.shape for tensor in tensors]
    for i in range(len(shapes)):
        if shapes[0] != shapes[i]:
            raise ValueError(f"fail shapes in {name}")


def _broadcast_shapes_with_name(name: str, *shapes: tuple[int, ...]) -> tuple[int, ...]:
    try:
        shape = broadcast_shapes(*shapes)
    except ValueError as e:
        raise ValueError(f"{name}: {e}")
    return shape


def to_symbolic(x: BaseTensor) -> SymbolicTensor:
    if isinstance(x, SymbolicTensor):
        return x
    if isinstance(x, ConcreteTensor):
        return Constant(x)
    raise TypeError(f"fail convert to symbolic: {x!r}")


def to_concrete(
    x: SymbolicTensor,
    buffer: ConcreteBuffer,
    event: DeviceEvent | None = None
) -> ConcreteTensor:
    return ConcreteTensor(
        x.shape,
        x.strides,
        x.buffer_offset,
        x.dtype,
        buffer,
        event=event
    )


def array(
    object: Any,
    dtype: DTypeLike | None = None, *,
    device: Device | None = None
) -> BaseTensor:
    if device is None:
        device = dannet.current_device()

    with device:
        if isinstance(object, SymbolicTensor):
            return object.astype(dtype)
        if isinstance(object, ConcreteTensor):
            if object.buffer.device != device:
                raise NotImplementedError()
            return object.astype(dtype)

        array = np.array(object, dtype=dtype)
        return ConcreteTensor.from_ndarray(array)


_slice_slice_info_type = tuple[
    tuple[slice, ...],
    tuple[int, ...],
    tuple[int, ...]
]
_slice_tuple_info_type = tuple[
    tuple[tuple[int, int, int], ...],
    tuple[int, ...],
    tuple[int, ...]
]


def parse_getitem_key(
    x: BaseTensor,
    key: _getitem_key_type
) -> _slice_slice_info_type:
    if not isinstance(key, tuple):
        key = (key,)

    n_ellipsis = sum(1 for k in key if k is Ellipsis)
    if n_ellipsis > 1:
        raise IndexError(
            'an index can only have a single ellipsis (\'...\')'
        )

    n_newaxes = sum(1 for k in key if k is None)
    full_key_len = x.ndim + n_newaxes
    missing = full_key_len - (len(key) - 1)

    key_norm: list[int | slice | None] = []
    for k in key:
        if isinstance(k, EllipsisType):
            key_norm.extend([slice(None)] * missing)
        else:
            key_norm.append(k)

    if len(key) < full_key_len:
        key = key + (slice(None),) * (full_key_len - len(key))

    for i, k in enumerate(key_norm):
        if not isinstance(k, (int, slice)) and k is not None:
            raise TypeError(
                f'Invalid index at position {i}: '
                f'expected int, slice, or None, got {type(k).__name__}'
            )

    slices: list[slice] = []
    squeeze_axes: list[int] = []
    newaxes_positions: list[int] = []

    for i, k in enumerate(key_norm):
        if k is None:
            newaxes_positions.append(i)
        elif isinstance(k, int):
            slices.append(slice(k, k + 1, 1))
            squeeze_axes.append(i)
        else:
            slices.append(k)
    return tuple(slices), tuple(newaxes_positions), tuple(squeeze_axes)


def normalize_slices(
    slices: list[tuple[int | None, int | None, int | None]],
    orig_shape: tuple[int, ...],
    orig_strides: tuple[int, ...]
) -> _slice_tuple_info_type:
    import math
    assert len(slices) == len(orig_strides) == len(orig_shape)
    new_shape: list[int] = []
    new_strides: list[int] = []
    new_slices: list[tuple[int, int, int]] = []
    for i, (start, stop, step) in enumerate(slices):
        dim = orig_shape[i]
        stride = orig_strides[i]

        if step is None:
            step = 1
        if step == 0:
            raise ValueError('slice step cannot be zero')
        if step < 0:
            default_start = dim - 1
            default_stop = -1
        else:
            default_start = 0
            default_stop = dim

        if start is None:
            start = default_start
        elif start < 0:
            start += dim
        if stop is None:
            stop = default_stop
        elif stop < 0:
            stop += dim

        if step < 0:
            start = max(0, min(start, dim - 1))
            stop = max(-1, min(stop, dim - 1))
            length = max(0, math.ceil((stop - start) / step))
        else:
            start = max(0, min(start, dim))
            stop = max(0, min(stop, dim))
            length = max(0, math.ceil((start - stop) / (-step)))
        new_shape.append(length)
        new_strides.append(stride * step)
        new_slices.append((start, stop, step))
    return tuple(new_slices), tuple(new_shape), tuple(new_strides)


_PyScalar: TypeAlias = bool | int | float | complex
TensorLike: TypeAlias = _PyScalar | BaseTensor | np.ndarray
