import builtins

import numpy as np
import dannet as dt

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.symbolic_scope import SymbolicScope

SUPPORTS_SPARSE_TENSORS = False
SUPPORTS_RAGGED_TENSORS = False
IS_THREAD_SAFE = False


class Variable(KerasVariable, dt.core.Variable):
    def _initialize(self, value):
        if isinstance(value, dt.core.Variable):
            self._value = value
        else:
            value = convert_to_tensor(value)
            value = dt.eval(value)
            self._value = dt.core.Variable(value, value.dtype)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype)

    def _direct_assign(self, value):
        self._value.assign(value)


def convert_to_tensor(x, dtype=None, sparse=None, ragged=None):
    if sparse:
        raise ValueError('`sparse=True` is not supported with dannet backend')
    if ragged:
        raise ValueError('`ragged=True` is not supported with dannet backend')
    if isinstance(x, Variable):
        return x.value

    if is_tensor(x):
        if dtype is not None and x.dtype != dtype:
            return dt.cast(x, dtype)
        return x

    x = dt.convert_to_tensor(x)
    if dtype is not None:
        x = cast(x, dtype)
    return x


def convert_to_numpy(x):
    if is_tensor(x):
        if isinstance(x, Variable):
            x = x._value
        x = dt.eval(x)
    if isinstance(x, (list, tuple)):
        return np.array([np.array(e) for e in x])
    return np.array(x)


def is_tensor(x):
    return isinstance(x, dt.core.TensorBase)


def shape(x):
    return x.shape


def cast(x, dtype):
    if isinstance(x, Variable):
        x = x.value
    if is_tensor(x):
        return dt.cast(x, dtype)
    return convert_to_tensor(x, dtype)


def compute_output_spec(fn, *args, **kwargs):
    def has_none_shape(x):
        if isinstance(x, KerasTensor):
            return None in x.shape
        return False

    def convert_keras_tensor_to_dannet(x, fill_value=None):
        if isinstance(x, KerasTensor):
            shape = list(x.shape)
            if fill_value:
                for i, e in enumerate(shape):
                    if e is None:
                        shape[i] = fill_value
            return dt.ones(shape, x.dtype)
        return x

    def convert_dannet_to_keras_tensor(x):
        if is_tensor(x) or isinstance(x, dt.compiler.OutputTensor):
            return KerasTensor(x.shape, standardize_dtype(x.dtype))
        return x

    def symbolic_call(fn, args, kwargs, fill_value):
        fn = dt.function*fn
        try:
            meta_args, meta_kwargs = tree.map_structure(
                lambda x: convert_keras_tensor_to_dannet(x, fill_value),
                (args, kwargs),
            )
            return fn.compute_output_spec(*meta_args, **meta_kwargs)
        except Exception:
            eager_args, eager_kwargs = tree.map_structure(
                lambda x: convert_keras_tensor_to_dannet(x, fill_value),
                (args, kwargs),
            )
            return fn.compute_output_spec(*eager_args, **eager_kwargs)

    with StatelessScope(), SymbolicScope():
        outputs = symbolic_call(fn, args, kwargs, fill_value=83)

        none_in_shape = any(
            builtins.map(has_none_shape, tree.flatten((args, kwargs)))
        )
        if none_in_shape:
            outputs_1 = outputs
            outputs_2 = symbolic_call(fn, args, kwargs, fill_value=89)

            flat_out_1 = tree.flatten(outputs_1)
            flat_out_2 = tree.flatten(outputs_2)

            flat_out = []
            for x1, x2 in zip(flat_out_1, flat_out_2):
                shape = list(x1.shape)
                for i, e in enumerate(x2.shape):
                    if e != shape[i]:
                        shape[i] = None
                flat_out.append(KerasTensor(
                    shape, standardize_dtype(x1.dtype)))
            outputs = tree.pack_sequence_as(outputs_1, flat_out)

        output_spec = tree.map_structure(
            convert_dannet_to_keras_tensor, outputs)

    return output_spec


def cond(pred, true_fn, false_fn):
    raise NotImplementedError


def vectorized_map(function, elements):
    raise NotImplementedError


def map(f, xs):
    raise NotImplementedError


def scan(f, init, xs=None, length=None, reverse=False, unroll=1):
    raise NotImplementedError


def associative_scan(f, elems, reverse=False, axis=0):
    raise NotImplementedError


def scatter(indices, values, shape):
    raise NotImplementedError


def slice_update(inputs, start_indices, updates):
    raise NotImplementedError


def switch(index, branches, *operands):
    raise NotImplementedError


def while_loop(
    cond,
    body,
    loop_vars,
    maximum_iterations=None,
):
    raise NotImplementedError


def fori_loop(lower, upper, body_fun, init_val):
    raise NotImplementedError


def stop_gradient(variable):
    raise NotImplementedError


def unstack(x, num=None, axis=0):
    raise NotImplementedError


def random_seed_dtype():
    return 'int32'


def remat(f):
    raise NotImplementedError
