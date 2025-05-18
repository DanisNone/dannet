import dannet as dt
from dannet.ops.math import (
    _ElementWiseUnary, _make_unary,
    _ElementWiseBinary, _make_binary
)


class _LogicalUnary(_ElementWiseUnary):
    def result_dtype(self, dtype):
        return dt.dtype.bool_dtype

    def _compute_gradients(self, grad):
        return None


class _LogicalBinary(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.bool_dtype

    def _compute_gradients(self, grad):
        return None


class _Equal(_LogicalBinary):
    pass


class _NotEqual(_LogicalBinary):
    pass


class _Greater(_LogicalBinary):
    pass


class _GreaterEqual(_LogicalBinary):
    pass


class _Less(_LogicalBinary):
    pass


class _LessEqual(_LogicalBinary):
    pass


class _LogicalNot(_LogicalUnary):
    pass


class _LogicalOr(_LogicalBinary):
    pass


class _LogicalAnd(_LogicalBinary):
    pass


class _LogicalXor(_LogicalBinary):
    pass


equal = _make_binary('equal', _Equal)
not_equal = _make_binary('not_equal', _NotEqual)
greater = _make_binary('greater', _Greater)
greater_equal = _make_binary('greater_equal', _GreaterEqual)
less = _make_binary('less', _Less)
less_equal = _make_binary('less_equal', _LessEqual)


logical_not = _make_unary('logical_not', _LogicalNot)
logical_or = _make_binary('logical_or', _LogicalOr)
logical_and = _make_binary('logical_and', _LogicalAnd)
logical_xor = _make_binary('logical_xor', _LogicalXor)


__all__ = [
    'equal',
    'not_equal',
    'greater',
    'greater_equal',
    'less',
    'less_equal',

    'logical_not',
    'logical_and',
    'logical_or',
    'logical_xor',
]
