import pytest

import numpy as np
import dannet as dt
from .utils import*

einsum_cases = [
    ('i,i->', lambda a, b: np.dot(a, b)),                   # dot product
    ('ij,j->i', lambda a, b: np.matmul(a, b)),              # matrix-vector
    ('ij,jk->ik', lambda a, b: np.matmul(a, b)),            # matrix-matrix
    ('bij,bjk->bik', lambda a, b: np.matmul(a, b)),         # batch matmul
    ('...ij,...jk->...ik', lambda a, b: np.matmul(a, b)),   # general broadcasting
    ('ij->ji', lambda a: np.swapaxes(a, -1, -2)),           # transpose
    ('ij->', lambda a: np.sum(a)),                          # sum all
    ('ijk,ijl->kl', lambda a, b: np.einsum('ijk,ijl->kl', a, b)),  # reduce over i and j
]


@pytest.mark.parametrize("einsum_eq,expected_fn", einsum_cases)
@pytest.mark.parametrize("dtype", dtypes)
@ensure_supported
def test_einsum(device, einsum_eq, expected_fn, dtype):
    # Generate input shapes based on the equation
    letters = set(einsum_eq.replace(',', '').replace('->', ''))
    dims = {letter: np.random.randint(1, 6) for letter in letters}

    def get_shape(term):
        return tuple(dims[ch] for ch in term)

    inputs = einsum_eq.split('->')[0].split(',')
    arrays = [random_array(get_shape(term), dtype) for term in inputs]

    with device:
        tensors = [dt.constant(arr) for arr in arrays]
        out = dt.einsum(einsum_eq, *tensors)
        out_np = out.numpy()

    try:
        expected = np.einsum(einsum_eq, *arrays)
    except ValueError:
        return

    equal_output(expected, out_np)
