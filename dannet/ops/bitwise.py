import dannet as dt
from dannet.ops.math import (
    _ElementWiseUnary, _make_unary,
    _ElementWiseBinary, _make_binary
)


class _BitwiseUnary(_ElementWiseUnary):
    def result_dtype(self, dtype):
        if dt.dtype.is_float_dtype(dtype) or dt.dtype.is_complex_dtype(dtype):
            raise TypeError(
                'Bitwise operations waiting integer or bool dtype.'
            )
        return dtype

    def _compute_gradients(self, grad):
        return None


class _BitwiseBinary(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if (
            dt.dtype.is_float_dtype(dtype1) or
            dt.dtype.is_float_dtype(dtype2) or
            dt.dtype.is_complex_dtype(dtype1) or
            dt.dtype.is_complex_dtype(dtype2)
        ):
            raise TypeError(
                'Bitwise operations waiting integer or bool dtype.'
            )
        result = dt.dtype.promote_dtypes(dtype1, dtype2)
        if dt.dtype.is_float_dtype(result):
            raise TypeError(
                'No suitable integer type to represent the result '
                'of the bitwise operation.'
            )
        return result

    def _compute_gradients(self, grad):
        return None


class _BitwiseNot(_BitwiseUnary):
    pass


class _BitwiseOr(_BitwiseBinary):
    pass


class _BitwiseAnd(_BitwiseBinary):
    pass


class _BitwiseXor(_BitwiseBinary):
    pass


class _Shift(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        if dt.dtype.is_bool_dtype(dtype1) and dt.dtype.is_bool_dtype(dtype2):
            return 'int32'

        if (
            dt.dtype.is_float_dtype(dtype1) or
            dt.dtype.is_float_dtype(dtype2) or
            dt.dtype.is_complex_dtype(dtype1) or
            dt.dtype.is_complex_dtype(dtype2)
        ):
            raise TypeError('shift operations waiting integer or bool dtype.')
        result = dt.dtype.promote_dtypes(dtype1, dtype2)
        if dt.dtype.is_float_dtype(result):
            raise TypeError(
                'No suitable integer type to represent the result '
                'of the shift operation.'
            )
        return result

    def _compute_gradients(self, grad):
        return None


class _RightShift(_Shift):
    pass


class _LeftShift(_Shift):
    pass


bitwise_not = _make_unary('bitwise_not', _BitwiseNot)
bitwise_invert = bitwise_not

bitwise_or = _make_binary('bitwise_or', _BitwiseOr)
bitwise_and = _make_binary('bitwise_and', _BitwiseAnd)
bitwise_xor = _make_binary('bitwise_xor', _BitwiseXor)

left_shift = _make_binary('left_shift', _LeftShift)
right_shift = _make_binary('right_shift', _RightShift)

__all__ = [
    'bitwise_not',
    'bitwise_invert',
    'bitwise_or',
    'bitwise_and',
    'bitwise_xor',

    'left_shift',
    'right_shift'
]
