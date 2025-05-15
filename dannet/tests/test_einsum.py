import pytest

import numpy as np
import dannet as dt
from .utils import*


dtypes_without_f16 = set(dtypes) - {'float16'}
einsum_cases = [
    'i,i->',
    'ij,j->i',
    'ij,jk->ik',
    'bij,bjk->bik',
    'qweij,qwejk->qweik',
    'ij->ji',
    'ij->',
    'ijk,ijl->kl'
]

@pytest.mark.parametrize("einsum_eq", einsum_cases)
@pytest.mark.parametrize("dtype", dtypes_without_f16)
@ensure_supported
def test_einsum(device, einsum_eq, dtype):
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
