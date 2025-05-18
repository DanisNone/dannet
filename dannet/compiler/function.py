from __future__ import annotations
from typing import Any

import tree

import numpy as np
import dannet as dt
import dannet.topsort as topsort

from dannet.graph_collections import GList


class InputTensor:
    def __init__(self, shape, dtype):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = dt.dtype.normalize_dtype(dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __eq__(self, other):
        if not isinstance(other, InputTensor):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self):
        return hash((self.shape, self.dtype))


class OutputTensor:
    def __init__(self, shape, dtype):
        self._shape = dt.utils.normalize_shape(shape)
        self._dtype = dt.dtype.normalize_dtype(dtype)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __eq__(self, other):
        if not isinstance(other, OutputTensor):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    def __hash__(self):
        return hash((self.shape, self.dtype))


class function:
    _run_instance: function | None = None

    def __init__(self, func):
        self._func = func
        self._nodes: GList[dt.core.TensorBase] = GList()

        self._cached: dict[Any, tuple] = {}

    def compute_output_spec(self, *args, **kwargs):
        if self._nodes:
            print(self._nodes)
            raise RuntimeError(
                'Internal node buffer not empty before graph build'
            )
        flatten_input, input_spec, inputs, idx = self._prepare_inputs(
            *args, **kwargs)
        flatten_input, placeholders = self._create_placeholders(
            flatten_input, idx)

        args_t: tuple
        kwargs_t: dict
        args_t, kwargs_t = tree.unflatten_as(
            (args, kwargs), flatten_input
        )  # type: ignore

        function._run_instance = self
        try:
            output = self._func(*args_t, **kwargs_t)
        except Exception as e:
            self._nodes.clear()
            raise e
        finally:
            function._run_instance = None

        return tree.map_structure(_get_output_spec, output)

    def _prepare_inputs(self, *args, **kwargs):
        flatten_input: list = tree.flatten((args, kwargs))
        flatten_input = tree.map_structure(
            _to_dannet_tensor, flatten_input
        )  # type: ignore
        input_spec = tuple(tree.map_structure(
            _get_input_spec, flatten_input
        ))  # type: ignore

        inputs: list[dt.core.Constant] = []
        idx: list[int] = []
        for i, obj in enumerate(flatten_input):
            if not isinstance(obj, dt.core.TensorBase):
                continue
            if dt.core._is_constant(obj):
                inputs.append(dt.eval(obj))
                idx.append(i)
            else:
                raise NotImplementedError(
                    'TensorBase inputs not supported yet'
                )

        return flatten_input, input_spec, inputs, idx

    def _create_placeholders(self, flatten_input: list, idx: list[int]):
        flatten_input = flatten_input.copy()
        placeholders = []
        for i in idx:
            placeholder = dt.core.Placeholder(
                flatten_input[i].shape, flatten_input[i].dtype
            )
            flatten_input[i] = placeholder
            placeholders.append(placeholder)
        return flatten_input, placeholders

    def __call__(self, *args, **kwargs):
        if function._run_instance is not None:
            raise NotImplementedError(
                'Nested function calls are not supported'
            )

        if self._nodes:
            raise RuntimeError(
                'Internal node buffer not empty before graph build'
            )

        flatten_input, input_spec, inputs, idx = self._prepare_inputs(
            *args, **kwargs)

        if input_spec in self._cached:
            (
                compiled,
                output,
                output_indexes,
                output_template
            ) = self._cached[input_spec]

            output_flatten = list(output_template)
            result_vals = compiled(inputs)
            for idx, val in zip(output_indexes, result_vals):
                output_flatten[idx] = val
            return tree.unflatten_as(output, output_flatten)

        flatten_input, placeholders = self._create_placeholders(
            flatten_input, idx
        )

        args_t: tuple
        kwargs_t: dict
        args_t, kwargs_t = tree.unflatten_as(
            (args, kwargs), flatten_input
        )  # type: ignore

        function._run_instance = self

        try:
            output = self._func(*args_t, **kwargs_t)
        except Exception as e:
            self._nodes.clear()
            raise e
        finally:
            function._run_instance = None

        flatten_output: list = tree.flatten(output)

        outputs: list[dt.core.TensorBase] = []
        output_indexes: list[int] = []
        for i, out in enumerate(flatten_output):
            if isinstance(out, dt.core.TensorBase):
                outputs.append(out)
                output_indexes.append(i)

        compiled = dt.compiler.compile(
            placeholders, outputs, self._nodes, is_eager_mode=False
        )
        self._nodes.clear()

        output_template: list[Any] = []
        for val in flatten_output:
            if isinstance(val, dt.core.TensorBase):
                output_template.append(None)
            else:
                output_template.append(val)

        self._cached[input_spec] = (
            compiled, output, output_indexes, output_template)

        result_vals = compiled(inputs)
        for idx, val in zip(output_indexes, result_vals):
            flatten_output[idx] = val
        return tree.unflatten_as(output, flatten_output)

    @classmethod
    def _add_node(cls, node: dt.core.TensorBase):
        self = cls._run_instance
        if self is None:
            raise RuntimeError(
                'Tensor operations must be used only in dt.function')

        input_tensors = (
            dt.core.Placeholder,
            dt.core.Variable,
            dt.core.Constant
        )
        for inp in node.inputs():
            if inp in self._nodes:
                continue
            if isinstance(inp, input_tensors):
                self._nodes.append(inp)
            else:
                raise RuntimeError(f'node {node} have unknown input: {inp}')
        if node not in self._nodes:
            self._nodes.append(node)


def _to_dannet_tensor(x):
    if isinstance(x, dt.core.Constant):
        return x
    if isinstance(x, np.ndarray):
        return dt.constant(x)
    if isinstance(x, dt.core.Variable):
        return x
    return x


def _get_input_spec(x):
    if isinstance(x, dt.core.TensorBase):
        return InputTensor(x.shape, x.dtype)
    return x


def _get_output_spec(x):
    if isinstance(x, dt.core.TensorBase):
        return OutputTensor(x.shape, x.dtype)
    return x


def is_eager():
    return function._run_instance is None


def eval(x: dt.typing.TensorLike) -> dt.core.Constant:
    x = dt.convert_to_tensor(x)
    if isinstance(x, dt.core.Constant):
        return x

    nodes = topsort.topological_sort([x])
    return dt.compiler.compile([], [x], nodes[::-1], is_eager_mode=True)([])[0]
