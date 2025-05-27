import os
import pathlib
from typing import Callable, Sequence

import pyopencl as cl
import numpy as np

import dannet as dt
from dannet.topsort import topological_sort
from dannet.device import DeviceBuffer, mem_flags

from dannet.core import TensorBase
from dannet.graph_collections import (
    GList, GDict, GSet
)


class NotSupportDtypeError(Exception):
    pass


class compile:
    _compile_uid: int = 0
    _log_dir_path: pathlib.Path | None = None

    def __init__(
        self,
        inputs: list[dt.core.Placeholder],
        outputs: list[TensorBase],
        nodes: list[TensorBase] | GList[TensorBase],
        is_eager_mode: bool
    ):
        self.device = dt.current_device()

        if not all(isinstance(inp, dt.core.Placeholder) for inp in inputs):
            raise TypeError('All inputs must be Placeholder instances.')

        if not all(isinstance(out, TensorBase) for out in outputs):
            raise TypeError('All outputs must be TensorBase instances.')

        if not all(isinstance(node, TensorBase) for node in nodes):
            raise TypeError('All nodes must be TensorBase instances.')

        self._inputs = GList(inputs)
        self._outputs = GList(outputs)
        for i in range(len(self._outputs)):
            if not self._outputs[i]._is_contiguous:
                self._outputs[i] = dt.basic._Copy(self._outputs[i])
                nodes.append(self._outputs[i])

        self._nodes = GList(nodes)
        self._constant_nodes: GList[dt.core.Constant] = GList()
        self._variable_nodes: GList[dt.core.Variable] = GList()
        self._compute_nodes: GList[TensorBase] = GList()

        self._buffers: GDict[dt.core.TensorBuffer, DeviceBuffer] = GDict()
        self._constants_loaded: bool = False
        self._kernels: list[tuple[
            TensorBase,
            Callable[[], cl.Event]
        ]] = []

        self._is_eager_mode = bool(is_eager_mode)
        if not self._is_eager_mode:
            self._remove_dead_code()
            self._sort_nodes()

        self._filter_nodes()

        if not self._is_eager_mode:
            self._write_nodes_info()
        self._allocate_buffers()
        self._compile_kernels()

        self._load_constants()

    def _remove_dead_code(self):
        target_tensors = self._inputs + self._outputs

        for node in self._nodes:
            if isinstance(node, dt.core.Update):
                target_tensors.append(node)

        nodes = GList(topological_sort(target_tensors))
        self._nodes = [node for node in self._nodes if node in nodes]

    def _sort_nodes(self):
        def sort_key(node):
            return max([node_indices[inp] for inp in node.inputs()], default=0)

        dependencies: GDict[TensorBase, GSet[TensorBase]] = GDict()
        update_dependencies: GDict[TensorBase, GSet[TensorBase]] = GDict()

        for node in self._nodes:
            dependencies[node] = GSet(node.inputs())
            for inp in node.inputs():
                dependencies[node] |= update_dependencies[inp]
            update_dependencies[node] = dependencies[node].copy()

            if isinstance(node, dt.core.Update):
                update_dependencies[node._variable].add(node)

        in_degree = GDict(
            (node, len(dependencies[node]))
            for node in self._nodes
        )
        node_indices: GDict[TensorBase, int] = GDict()
        nodes: GList[TensorBase] = GList()

        ready = GList(node for node in self._nodes if in_degree[node] == 0)

        while ready:
            ready.sort(key=sort_key)

            for node in ready:
                node_indices[node] = len(nodes)
                nodes.append(node)

                for other in self._nodes:
                    if node in dependencies[other]:
                        in_degree[other] -= 1
                        dependencies[other].discard(node)

            ready = GList(
                node for node in self._nodes
                if (
                    in_degree[node] == 0 and
                    node not in node_indices
                )
            )

        self._nodes = nodes

    def _filter_nodes(self):
        for node in self._nodes:
            if isinstance(node, dt.core.Constant):
                self._constant_nodes.append(node)
            elif isinstance(node, dt.core.Variable):
                self._variable_nodes.append(node)
            elif isinstance(node, dt.core.Placeholder):
                if node not in self._inputs:
                    raise ValueError('graph have unknown input')
            else:
                self._compute_nodes.append(node)

    def _write_nodes_info(self):
        import csv
        compile._compile_uid += 1
        if self._log_dir_path is None:
            return
        file_path = self._log_dir_path / f'{compile._compile_uid}.csv'

        last_usage: GDict[dt.core.TensorBuffer, int] = GDict()
        for idx, node in enumerate(self._nodes):
            buf = node._buffer
            last_usage[buf] = idx

        unique_bufs = list(GSet(node._buffer for node in self._nodes))
        buf_id_map = GDict((buf, i) for i, buf in enumerate(unique_bufs))

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'node_id', 'op_type', 'category',
                'shape', 'dtype', 'strides',
                'inputs', 'buffer_id', 'buffer_bytes', 'last_used',
            ])

            for idx, node in enumerate(self._nodes):
                op_type = type(node).__name__
                if isinstance(node, dt.core.Placeholder):
                    category = 'Input'
                elif isinstance(node, dt.core.Constant):
                    category = 'Constant'
                elif isinstance(node, dt.core.Variable):
                    category = 'Variable'
                elif isinstance(node, dt.core.Update):
                    category = 'Update'
                else:
                    category = 'Compute'

                shape = node.shape
                dtype = node.dtype
                strides = node.strides

                input_ids = [self._nodes.index(inp) for inp in node.inputs()]

                buf = node._buffer
                buf_id = buf_id_map[buf]
                buf_bytes = buf.nbytes
                last_idx = last_usage[buf]

                writer.writerow([
                    idx, op_type, category,
                    shape, dtype, strides,
                    input_ids, buf_id, buf_bytes, last_idx
                ])

    def _get_free_buffer(
        self,
        allocated_buffers: list[DeviceBuffer],
        nbytes: int
    ) -> tuple[DeviceBuffer, bool]:
        for buffer in allocated_buffers:
            if buffer.nbytes == nbytes:
                return (buffer, True)
        buffer = self.device.allocate_buffer(mem_flags.READ_WRITE, nbytes)
        return (buffer, False)

    def _allocate_buffers(self):
        need_buffers: list[dt.core.TensorBuffer] = []
        for node in self._nodes:
            if node._buffer not in need_buffers:
                need_buffers.append(node._buffer)

        buffer_usage: GDict[dt.core.TensorBuffer, int] = GDict()
        for buffer in need_buffers:
            if buffer not in buffer_usage:
                buffer_usage[buffer] = 0

            for inp in buffer.inputs():
                buffer_usage[inp] += 1

        for node in self._variable_nodes + self._constant_nodes:
            buffer_usage[node._buffer] = -1

        free_buffer: list[DeviceBuffer] = []

        for buffer in need_buffers:
            device_buffer, is_reused = self._get_free_buffer(
                free_buffer, buffer.nbytes
            )
            if is_reused:
                free_buffer.remove(device_buffer)

            self._buffers[buffer] = device_buffer

            for inp in buffer.inputs():
                buffer_usage[inp] -= 1
                if buffer_usage[inp] == 0:
                    free_buffer.append(self._buffers[inp])

    def _load_constants(self):
        if self._constants_loaded:
            return

        with dt.timestat.record('load_constants'):
            for const in self._constant_nodes:
                value = const.numpy()
                buffer = self._buffers[const._buffer]

                assert value.nbytes == buffer.nbytes
                self.device.enqueue_copy(buffer, value)
            self._constants_loaded = True
            self.device.queue.finish()

    def _load_variables(self):
        with dt.timestat.record('load_variables'):
            for var in self._variable_nodes:
                if var._used_by == self:
                    continue

                value = var.numpy()
                var._used_by = self

                buffer = self._buffers[var._buffer]
                assert value.nbytes == buffer.nbytes
                self.device.enqueue_copy(buffer, value)
            self.device.queue.finish()

    def _load_inputs(self, data: Sequence[np.ndarray | dt.core.Constant]):
        if len(self._inputs) != len(data):
            raise ValueError(
                f'Expected {len(self._inputs)} input(s), but got {len(data)}.')

        with dt.timestat.record('load_inputs'):
            for inp, array in zip(self._inputs, data):
                if inp.shape != array.shape:
                    raise ValueError(
                        f'Shape mismatch for input {inp}: '
                        f'expected {inp.shape}, got {array.shape}'
                    )
                if isinstance(array, dt.core.Constant):
                    array = array._value

                array = array.astype(inp.dtype).copy()
                buffer = self._buffers[inp._buffer]
                self.device.enqueue_copy(buffer, array)
            self.device.queue.finish()

    def _get_results(self) -> list[dt.core.Constant]:
        result = []
        with dt.timestat.record('get_results'):
            for out in self._outputs:
                res = np.empty(out.shape, dtype=out.dtype)
                buffer = self._buffers[out._buffer]
                self.device.enqueue_copy(res, buffer, is_blocking=True)
                result.append(dt.constant(res))
            self.device.queue.finish()

        return result

    def _get_variable(self, var: dt.core.Variable) -> np.ndarray:
        if var not in self._variable_nodes:
            raise RuntimeError(
                f'Variable {var} is not part of this Function graph.')

        value = np.empty(var.shape, dtype=var.dtype)
        buffer = self._buffers[var._buffer]

        self.device.enqueue_copy(value, buffer)
        return value

    def _compile_kernels(self):
        if self._kernels:
            return
        with dt.timestat.record('compile_kernels'):
            for node in self._compute_nodes:
                input_buffers = [
                    self._buffers[inp._buffer]
                    for inp in node.inputs()
                ]
                output_buffer = self._buffers[node._buffer]
                kernel = dt.compiler.compile_node(
                    self.device, node, input_buffers, output_buffer
                )
                if kernel is not None:
                    self._kernels.append((node, kernel))

    def __call__(
        self,
        data: Sequence[np.ndarray | dt.core.Constant]
    ) -> list[dt.core.Constant]:
        with dt.timestat.record('call'):
            self._load_variables()
            self._load_inputs(data)

            events: GDict[TensorBase, cl.Event] = GDict()
            with dt.timestat.record('execute_kernels'):
                if dt.timestat.enabled():
                    for node, kernel in self._kernels:
                        s = str([(inp.shape, inp.dtype)
                                for inp in node.inputs()])
                        s = f'{node}: {s}'
                        with dt.timestat.record(s):
                            kernel()
                            self.device.queue.finish()

                else:
                    for node, kernel in self._kernels:
                        for inp in node.inputs():
                            if inp in events:
                                events.pop(inp).wait()

                        events[node] = kernel()

                for event in events.values():
                    event.wait()
            return self._get_results()


def set_node_logging_dir(path: str | os.PathLike):
    path = pathlib.Path(path)
    if not path.is_dir():
        raise NotADirectoryError(f'{path} is not a valid directory.')

    compile._log_dir_path = path


def disable_node_logging():
    compile._log_dir_path = None
