from __future__ import annotations

import abc
import math
from typing import Any, Sequence
import numpy as np

import dannet as dt


class Buffer:
    def __init__(self, parent: TensorBase):
        self.nbytes = parent.nbytes
        self.parent = parent

        if self.nbytes <= 0:
            raise ValueError('Buffer size must be greater than 0.')

        if not isinstance(self.parent, TensorBase):
            raise TypeError('Parent must be an instance of TensorBase.')

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Buffer):
            return False
        return self.nbytes == other.nbytes and self.parent == other.parent

    def __hash__(self):
        return hash((self.nbytes, self.parent))


class TensorBase(abc.ABC):
    _dtype: str
    _shape: tuple[int, ...]
    _strides: tuple[int, ...]
    _buffer: Buffer
    _buffer_offset: int

    @abc.abstractmethod
    def inputs(self) -> list[TensorBase]:
        pass

    @abc.abstractmethod
    def compute_gradients(self, grad: TensorBase) -> Sequence[TensorBase]:
        pass

    def get_config(self) -> dict[str, Any]:
        return {'shape': self._shape, 'dtype': self._dtype}
    
    def __eq__(self, other):
        if self is other:
            return True
        
        if type(self) != type(other):
            return False
        
        if self.get_config() != other.get_config():
            return False
        return self.inputs() == other.inputs()
    
    def __hash__(self):
        if not hasattr(self, '_hash'):
            config = self.get_config()
            config = tuple(config.items())
            self._hash = hash((config, *self.inputs()))
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
        return tuple(math.prod(self._shape[i+1:]) for i in range(len(self._shape)))

    def _is_default_strides(self):
        return self._strides == self._default_strides()
    
    def __add__(self, other):
        return dt.math.add(self, other)

    def __sub__(self, other):
        return dt.math.subtract(self, other)

    def __mul__(self, other):
        return dt.math.multiply(self, other)

    def __truediv__(self, other):
        return dt.math.divide(self, other)

    def __radd__(self, other):
        return dt.math.add(other, self)

    def __rsub__(self, other):
        return dt.math.subtract(other, self)

    def __rmul__(self, other):
        return dt.math.multiply(other, self)

    def __rtruediv__(self, other):
        return dt.math.divide(other, self)

    def __neg__(self):
        return dt.math.negative(self)
    
    def __repr__(self):
        name = type(self).__name__
        name = "".join(f"_{c.lower()}" if c.isupper() else c for c in name)
        name = name.lstrip('_')
        return f"<{name} shape={self._shape} dtype={self._dtype}>"
    
    def __bool__(self):
        if not dt.is_eager():
            raise NotImplementedError("Boolean evaluation is only supported in eager mode.")
        if self.size != 1:
            raise ValueError("Only scalar tensors can be used as a boolean.")
        
        self = dt.reshape(self, ())
        self = dt.eval(self)
        return bool(self._value)
        

class Constant(TensorBase):
    def __init__(self, value: dt.typing.TensorLike, dtype: dt.typing.DTypeLike | None  = None):
        if isinstance(value, TensorBase):
            raise NotImplementedError()
        
        if isinstance(value, int) and dtype is None:
            dtype = dt.dtype.int_dtype
        if isinstance(value, float) and dtype is None:
            dtype = dt.dtype.float_dtype
        self._value = np.array(value, dtype=dtype)
        dtype = self._value.dtype
        
        self._dtype = dt.dtype.normalize_dtype(self._value.dtype)
        self._shape = dt.utils.normalize_shape(self._value.shape)

        self._buffer = Buffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()        
        

    def get_value(self):
        return self._value.copy()
    
    def inputs(self):
        return []
    
    def compute_gradients(self, grad):
        return []
    
    def get_config(self):
        config = super(Constant, self).get_config()
        config['value'] = self._value.copy()
        return config

    def __eq__(self, other):
        if not isinstance(other, Constant):
            return False
        if self._shape != other._shape:
            return False
        if self._dtype != other._dtype:
            return False
        if not np.all(self._value == other._value):
            return False
        return True
    
    def __hash__(self):
        return hash((Constant, self._shape, self._dtype))
    
    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self._dtype
        return self._value.astype(dtype, copy=True)
    
    def __repr__(self):
        v = str(self._value)
        if len(v) > 50:
            v = v[:50] + "..."
        return f"Constant(shape={self._shape}, dtype={self._dtype}, numpy={v})"

class Variable(TensorBase):
    def __init__(self, value: dt.typing.TensorLike, dtype: dt.typing.DTypeLike | None = None):
        if isinstance(value, Constant):
            value = value.get_value()
        elif isinstance(value, TensorBase):
            raise NotImplementedError()
        
        self._value = np.array(value, dtype=dtype)
        dtype = self._value.dtype
        
        self._dtype = dt.dtype.normalize_dtype(self._value.dtype)
        self._shape = dt.utils.normalize_shape(self._value.shape)

        self._buffer = Buffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()     

        self._used_by: dt.compiler.compile | None = None
    
    def get_value(self):
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
        return dt.core._node_prepare(Update(self, value))

    def inputs(self):
        return []
    
    def compute_gradients(self, grad):
        return []

    def __eq__(self, other):
        return self is other
    
    def __hash__(self):
        return id(self)

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self._dtype
        self._used_by = None
        return self._value.astype(dtype, copy=True)
    
    def __repr__(self):
        v = str(self._value)
        if len(v) > 50:
            v = v[:50] + "..."
        return f"Constant(shape={self._shape}, dtype={self._dtype}, numpy={v})"

    
class Placeholder(TensorBase):
    def __init__(self, shape: dt.typing.ShapeLike, dtype: dt.typing.DTypeLike):
        self._dtype = dt.dtype.normalize_dtype(dtype)
        self._shape = dt.utils.normalize_shape(shape)

        self._buffer = Buffer(self)
        self._buffer_offset = 0
        self._strides = self._default_strides()        
        
    def inputs(self):
        return []
    
    def compute_gradients(self, grad):
        return []

    def __eq__(self, other):
        return self is other
    
    def __hash__(self):
        return id(self)

class Update(TensorBase):
    def __init__(self, variable: Variable, value: dt.typing.TensorLike):
        if not isinstance(variable, Variable):
            raise TypeError('Only Variable must be updated')
        
        self._variable = variable
        self._value = dt.convert_to_tensor(value)

        if self._variable._shape != self._value._shape:
            raise ValueError(f'Shape mismatch: variable shape {self._variable._shape} != value shape {self._value._shape}')

        self._value = dt.cast(self._value, self._variable._dtype)

        self._shape = self._variable._shape
        self._dtype = self._variable._dtype
        
        self._buffer = self._variable._buffer
        self._buffer_offset = self._variable._buffer_offset
        self._strides = self._variable._strides
        
    def inputs(self):
        return [self._variable, self._value]
    
    def compute_gradients(self, grad):
        raise TypeError('Update operation not have gradient')

    def __eq__(self, other):
        return self is other
    
    def __hash__(self):
        return id(self)


def _node_prepare(node: TensorBase):
    if dt.is_eager():
        return dt.eval(node)
    dt.function._add_node(node)
    return node