import dannet as dt
from dannet.ops.math import (
    _ElementWiseUnary, _make_unary,
    _ElementWiseBinary, _make_binary
)


class _BitwiseUnary(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_float_dtype(dtype):
            raise TypeError('Bitwise operations wait integer or bool dtype')
        return dtype

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x)]


class _BitwiseBinary(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if dt.dtype.is_float_dtype(dtype1):
            raise TypeError('Bitwise operations wait integer or bool dtype')
        if dt.dtype.is_float_dtype(dtype2):
            raise TypeError('Bitwise operations wait integer or bool dtype')
        return dt.dtype.max_dtype(dtype1, dtype2)

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x), dt.zeros_like(self.y)]


class _BitwiseNot(_BitwiseUnary):
    pass


class _BitwiseOr(_BitwiseBinary):
    pass


class _BitwiseAnd(_BitwiseBinary):
    pass


class _BitwiseXor(_BitwiseBinary):
    pass


bitwise_not = _make_unary('bitwise_not', _BitwiseNot)
bitwise_invert = bitwise_not

bitwise_or = _make_binary('bitwise_or', _BitwiseOr)
bitwise_and = _make_binary('bitwise_and', _BitwiseAnd)
bitwise_xor = _make_binary('bitwise_xor', _BitwiseXor)


__all__ = [
    'bitwise_not',
    'bitwise_invert',
    'bitwise_or',
    'bitwise_and',
    'bitwise_xor',
]
