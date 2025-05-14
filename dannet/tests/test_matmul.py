import pytest

import numpy as np
import dannet as dt
from .utils import*


@pytest.mark.parametrize("dtype", dtypes)
@pytest.mark.parametrize("shape_a,shape_b", [
    ((2,), (2,)),                 # dot
    ((3, 4), (4,)),               # matrix-vector
    ((3, 4), (4, 5)),             # matrix-matrix
    ((10, 3, 4), (10, 4, 5)),     # batch matmul
    ((2, 1, 3, 4), (2, 1, 4, 6)), # broadcasted batch matmul
])
@ensure_supported
def test_matmul(device, dtype, shape_a, shape_b):
    a_np = random_array(shape_a, dtype)
    b_np = random_array(shape_b, dtype)

    with device:
        a = dt.constant(a_np)
        b = dt.constant(b_np)
        c = dt.matmul(a, b)
        c_np = c.numpy()
    
    try:
        expected = np.matmul(a_np, b_np)
    except ValueError:
        return  # broadcasting or shape mismatch

    equal_output(expected, c_np)


