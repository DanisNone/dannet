from __future__ import annotations

import abc
import math
from typing import Any, Sequence
import numpy as np

import dannet as dt


class TensorMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)

        required_attrs = [
            '_buffer', '_buffer_offset',
            '_strides', '_is_contiguous'
        ]
        for attr in required_attrs:
            if not hasattr(instance, attr):
                raise AttributeError(
                    f'Missing required attribute \'{attr}\' '
                    f'in instance of {cls.__name__}'
                )

        return instance


class TensorBuffer:
    def __init__(self, parent: TensorBase):
        if not isinstance(parent, TensorBase):
            raise TypeError('Parent must be an instance of TensorBase.')

        self.parent = parent
        self.nbytes = self.parent.nbytes

    def __graph_eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, TensorBuffer):
            return False
        return self.parent.__graph_eq__(other.parent)

    def __graph_hash__(self):
        return hash((self.nbytes, self.parent.__graph_hash__()))

    def inputs(self) -> list[TensorBuffer]:
        return [inp._buffer for inp in self.parent.inputs()]


class TensorBase(abc.ABC, metaclass=TensorMeta):
    _dtype: str
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _buffer: TensorBuffer
    _buffer_offset: int
    _is_contiguous: bool

    @abc.abstractmethod
    def inputs(self) -> list[TensorBase]:
        pass

    @abc.abstractmethod
    def _compute_gradients(
        self,
        grad: TensorBase
    ) -> Sequence[TensorBase | None] | None:
        pass

    def compute_gradients(self, grad: TensorBase) -> list[TensorBase]:
        inputs = self.inputs()
        grads = self._compute_gradients(grad)
        if grads is None:
            grads = [None] * len(inputs)

        if len(grads) != len(inputs):
            raise ValueError(
                f"compute_gradients: expected {len(inputs)} gradient(s) "
                f"but got {len(grads)}"
            )

        result = []
        for i, (inp_grad, inp) in enumerate(zip(grads, inputs)):
            if inp_grad is None:
                inp_grad = dt.zeros_like(inp)

            if inp_grad.shape != inp.shape:
                raise ValueError(
                    f"compute_gradients: shape mismatch for input #{i}. "
                    f"expected {tuple(inp.shape)}, got {tuple(inp_grad.shape)}"
                )

            result.append(inp_grad)
        return result

    @abc.abstractmethod
    def get_config(self) -> dict[str, Any]:
        pass

    def numpy(self) -> np.ndarray:
        if _is_constant(self):
            return dt.eval(self)._value.copy()
        raise ValueError('Only constant tensor may be converted to numpy')

    def tolist(self):
        return self.numpy().tolist()

    def __array__(self, dtype=None):
        res = self.numpy()
        if dtype is not None:
            res = res.astype(dtype)
        return res

    def __graph_eq__(self, other):
        if self is other:
            return True
        if type(self) is not type(other):
            return False

        if self.shape != other.shape:
            return False
        if self.dtype != other.dtype:
            return False

        inps1 = self.inputs()
        inps2 = other.inputs()
        if len(inps1) != len(inps2):
            return False

        return all(
            inp1.__graph_eq__(inp2)
            for inp1, inp2 in zip(inps1, inps2)
        )

    def __graph_hash__(self):
        if not hasattr(self, '_hash'):
            config = self.get_config()
            config = tuple(config.items())

            hash_obj = [config, self.shape, self.dtype]
            for inp in self.inputs():
                hash_obj.append(inp.__graph_hash__())
            self._hash = hash(tuple(hash_obj))
        return self._hash

    @property
    def dtype(self) -> str:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def strides(self) -> tuple[int, ...]:
        return self._strides

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def itemsize(self) -> int:
        return dt.dtype.itemsize(self._dtype)

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.itemsize

    def _default_strides(self):
        return tuple(
            math.prod(self._shape[i+1:]) for i in range(len(self._shape))
        )

    def _init_default_buffer(self):
        self._buffer = TensorBuffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()
        self._is_contiguous = True

    def __add__(self, other):
        return dt.add(self, other)

    def __sub__(self, other):
        return dt.subtract(self, other)

    def __mul__(self, other):
        return dt.multiply(self, other)

    def __truediv__(self, other):
        return dt.divide(self, other)

    def __floordiv__(self, other):
        return dt.floor_divide(self, other)

    def __radd__(self, other):
        return dt.add(other, self)

    def __rsub__(self, other):
        return dt.subtract(other, self)

    def __rmul__(self, other):
        return dt.multiply(other, self)

    def __rtruediv__(self, other):
        return dt.divide(other, self)

    def __rfloordiv__(self, other):
        return dt.floor_divide(other, self)

    def __neg__(self):
        return dt.negative(self)

    def __eq__(self, other):  # type: ignore[override]
        return dt.equal(self, other)

    def __ne__(self, other):  # type: ignore[override]
        return dt.not_equal(self, other)

    def __lt__(self, other):
        return dt.less(self, other)

    def __le__(self, other):
        return dt.less_equal(self, other)

    def __gt__(self, other):
        return dt.greater(self, other)

    def __ge__(self, other):
        return dt.greater_equal(self, other)

    def __repr__(self):
        name = type(self).__name__
        name = ''.join(f'_{c.lower()}' if c.isupper() else c for c in name)
        name = name.lstrip('_')

        shape = getattr(self, '_shape', 'UNKNOWN')
        dtype = getattr(self, '_dtype', 'UNKNOWN')
        return f'<{name} shape={shape} dtype={dtype}>'

    def __bool__(self):
        if self.shape != ():
            raise ValueError(
                'Only scalar arrays can be converted to Python scalars.'
            )

        if not _is_constant(self) and not dt.is_eager():
            raise NotImplementedError(
                'Boolean evaluation is only supported in eager mode.'
            )
        return bool(dt.eval(self)._value)

    def __int__(self):
        if self.shape != ():
            raise ValueError(
                'Only scalar arrays can be converted to Python scalars.'
            )

        if not _is_constant(self) and not dt.is_eager():
            raise NotImplementedError(
                'Integer evaluation is only supported in eager mode.'
            )
        return int(dt.eval(self)._value)

    def __float__(self):
        if self.shape != ():
            raise ValueError(
                'Only scalar arrays can be converted to Python scalars.'
            )

        if not _is_constant(self) and not dt.is_eager():
            raise NotImplementedError(
                'Float evaluation is only supported in eager mode.'
            )
        return float(dt.eval(self)._value)

    def __getitem__(
        self,
        key: int | slice | None | tuple[int | slice | None, ...]
    ) -> TensorBase:
        if not isinstance(key, tuple):
            key = (key,)

        for i, k in enumerate(key):
            if not isinstance(k, (int, slice)) and k is not None:
                raise TypeError(
                    f'Invalid index at position {i}: '
                    f'expected int, slice, or None, got {type(k).__name__}'
                )
        n_newaxes = sum(1 for k in key if k is None)

        full_key_len = self.ndim + n_newaxes
        if len(key) < full_key_len:
            key += (slice(None),) * (full_key_len - len(key))

        slices: list[slice] = []
        squeeze_axes: list[int] = []
        newaxes_positions: list[int] = []

        for i, k in enumerate(key):
            if k is None:
                newaxes_positions.append(i)
            elif isinstance(k, int):
                slices.append(slice(k, k + 1, 1))
                squeeze_axes.append(i)
            elif isinstance(k, slice):
                slices.append(k)

        result = dt.slice(self, slices)

        for axis in newaxes_positions:
            result = dt.expand_dims(result, axis=axis)

        if squeeze_axes:
            result = dt.squeeze(result, axis=squeeze_axes)

        return result

    def __abs__(self):
        return dt.abs(self)

    def __invert__(self):
        return dt.bitwise_invert(self)

    def any(self, axis=None, keepdims=False):
        return dt.any(self, axis=axis, keepdims=keepdims)

    def all(self, axis=None, keepdims=False):
        return dt.all(self, axis=axis, keepdims=keepdims)

    def argmax(self, axis=None):
        return dt.argmax(self, axis=axis)

    def argmin(self, axis=None):
        return dt.argmin(self, axis=axis)

    def clip(self, min_val, max_val):
        return dt.clip(self, min_val, max_val)

    def cast(self, dtype: dt.typing.DTypeLike):
        return dt.cast(self, dtype)

    astype = cast

    def max(self, axis=None, keepdims=False):
        return dt.max(self, axis=axis, keepdims=keepdims)

    def mean(self, axis=None, keepdims=False):
        return dt.mean(self, axis=axis, keepdims=keepdims)

    def min(self, axis=None, keepdims=False):
        return dt.min(self, axis=axis, keepdims=keepdims)

    def prod(self, axis=None, keepdims=False):
        return dt.prod(self, axis=axis, keepdims=keepdims)

    def reshape(self, shape: dt.typing.ShapeLike):
        return dt.reshape(self, shape)

    def std(self, axis=None, keepdims=False):
        return dt.std(self, axis=axis, keepdims=keepdims)

    def sum(self, axis=None, keepdims=False):
        return dt.sum(self, axis=axis, keepdims=keepdims)

    def transpose(self, axes=None):
        return dt.transpose(self, axes=axes)

    def var(self, axis=None, keepdims=False):
        return dt.var(self, axis=axis, keepdims=keepdims)


class Constant(TensorBase):
    def __init__(
        self,
        value: dt.typing.TensorLike,
        dtype: dt.typing.DTypeLike | None = None
    ):
        if (
            not isinstance(value, np.generic) and
            dtype is None
        ):
            if isinstance(value, int):
                dtype = dt.dtype.int_dtype
            if isinstance(value, float):
                dtype = dt.dtype.float_dtype
        self._value = np.array(value, dtype=dtype)
        dtype = self._value.dtype

        self._dtype = dt.dtype.normalize_dtype(self._value.dtype)
        self._shape = dt.utils.normalize_shape(self._value.shape)

        self._init_default_buffer()

    def inputs(self):
        return []

    def _compute_gradients(self, grad):
        return []

    def __graph_eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Constant):
            return False
        if self.shape != other.shape or self.dtype != other.dtype:
            return False
        return np.array_equal(self._value, other._value, equal_nan=True)

    def __graph_hash__(self):
        return hash((Constant, self._shape, self._dtype))

    def __repr__(self):
        shape = getattr(self, '_shape', 'UNKNOWN')
        dtype = getattr(self, '_dtype', 'UNKNOWN')
        v = str(self._value)
        if len(v) > 50:
            v = v[:50] + '...'
        v = v.replace('\n', ' ')
        return f'<Constant(shape={shape}, dtype={dtype}, numpy={v})>'

    def get_config(self):
        return {}


class Variable(TensorBase):
    def __init__(
        self,
        value: dt.typing.TensorLike,
        dtype: dt.typing.DTypeLike | None = None
    ):
        if isinstance(value, Constant):
            value = value.numpy()
        elif isinstance(value, TensorBase):
            raise NotImplementedError()

        self._value = np.array(value, dtype=dtype)
        dtype = self._value.dtype

        self._dtype = dt.dtype.normalize_dtype(self._value.dtype)
        self._shape = dt.utils.normalize_shape(self._value.shape)

        self._init_default_buffer()
        self._used_by: dt.compiler.compile | None = None

    def numpy(self):
        if self._used_by is None:
            return self._value.copy()

        value = self._used_by._get_variable(self)
        assert self.shape == value.shape
        value = value.astype(self.dtype)

        np.copyto(self._value, value)
        self._used_by = None

        return self._value.copy()

    def assign(self, value):
        if dt.function._run_instance is None:
            value = np.asarray(value, self.dtype)
            if self.shape != value.shape:
                raise ValueError('')
            self._value = value.copy()
            self._used_by = None
            return
        dt.core._node_prepare(Update(self, value))

    def inputs(self):
        return []

    def _compute_gradients(self, grad):
        return []

    def __graph_eq__(self, other):
        return self is other

    def __graph_hash__(self):
        return id(self)

    def __array__(self, dtype=None):
        if not dt.is_eager():
            raise RuntimeError(
                'Variable to ndarray can be converted only in eager mode')

        if dtype is None:
            dtype = self._dtype
        self._used_by = None
        return self._value.astype(dtype, copy=True)

    def __repr__(self):
        shape = getattr(self, '_shape', 'UNKNOWN')
        dtype = getattr(self, '_dtype', 'UNKNOWN')
        v = str(self.numpy())
        if len(v) > 50:
            v = v[:50] + '...'
        v = v.replace('\n', ' ')
        return f'<Variable(shape={shape}, dtype={dtype}, numpy={v})>'

    def get_config(self):
        return {}


class Placeholder(TensorBase):
    def __init__(self, shape: dt.typing.ShapeLike, dtype: dt.typing.DTypeLike):
        self._dtype = dt.dtype.normalize_dtype(dtype)
        self._shape = dt.utils.normalize_shape(shape)

        self._init_default_buffer()

    def inputs(self):
        return []

    def _compute_gradients(self, grad):
        return []

    def __graph_eq__(self, other):
        return self is other

    def __graph_hash__(self):
        return id(self)

    def get_config(self):
        return {}


class Update(TensorBase):
    def __init__(self, variable: Variable, value: dt.typing.TensorLike):
        if not isinstance(variable, Variable):
            raise TypeError('Only Variable must be updated')

        self._variable = variable
        self._value = dt.convert_to_tensor(value)

        if self._variable._shape != self._value._shape:
            raise ValueError(
                f'Shape mismatch: '
                f'variable shape {self._variable._shape}'
                f'!= value shape {self._value._shape}'
            )

        self._value = dt.cast(self._value, self._variable._dtype)

        self._shape = self._variable._shape
        self._dtype = self._variable._dtype

        self._buffer = self._variable._buffer
        self._buffer_offset = self._variable._buffer_offset
        self._strides = self._variable._strides
        self._is_contiguous = self._value._is_contiguous

    def inputs(self):
        return [self._variable, self._value]

    def _compute_gradients(self, grad):
        raise TypeError('Update operation not have gradient')

    def __graph_eq__(self, other):
        return self is other

    def __graph_hash__(self):
        return id(self)

    def get_config(self):
        return {}


def _is_constant(node: TensorBase) -> bool:
    assert isinstance(node, TensorBase)
    if isinstance(node, Constant):
        return True
    inputs = node.inputs()
    if inputs:
        return all(_is_constant(inp) for inp in inputs)
    return False


def _node_prepare(node: TensorBase):
    if dt.is_eager() or _is_constant(node):
        return dt.eval(node)
    dt.function._add_node(node)
    return node


variable = Variable
constant = Constant
