import dannet as dt
from dannet.ops.math import _ElementWiseBinary
from dannet.ops.math import _make_binary


class _Logical(_ElementWiseBinary):
    def result_dtype(self, dtype1, dtype2):
        return dt.dtype.bool_dtype

    def compute_gradients(self, grad):
        return [dt.zeros_like(self.x), dt.zeros_like(self.y)]

class _Equal(_Logical):
    pass

class _NotEqual(_Logical):
    pass

class _Greater(_Logical):
    pass

class _GreaterEqual(_Logical):
    pass

class _Less(_Logical):
    pass

class _LessEqual(_Logical):
    pass

equal = _make_binary('equal', _Equal)
not_equal = _make_binary('not_equal', _NotEqual)
greater = _make_binary('greater', _Greater)
greater_equal = _make_binary('greater_equal', _GreaterEqual)
less = _make_binary('less', _Less)
less_equal = _make_binary('less_equal', _LessEqual)


__all__ = [
    'equal',
    'not_equal',
    'greater',
    'greater_equal',
    'less',
    'less_equal',
]