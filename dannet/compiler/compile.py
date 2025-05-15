import os
import pathlib
from typing import Callable, Sequence
from collections import OrderedDict

import pyopencl as cl
import numpy as np

import dannet as dt
from dannet.topsort import topological_sort

from dannet.core import TensorBase


class NotSupportDtypeError(BaseException):
    pass


class compile:
    _compile_uid: int = 0
    _log_dir_path: pathlib.Path | None = None
    
    def __init__(self,
        inputs: list[dt.core.Placeholder],
        outputs: list[TensorBase],
        nodes: list[TensorBase],
        is_eager_mode: bool
    ):
        self.device = dt.current_device()

        if not all(isinstance(inp, dt.core.Placeholder) for inp in inputs):
            raise TypeError('All inputs must be Placeholder instances.')
        
        if not all(isinstance(out, TensorBase) for out in outputs):
            raise TypeError('All outputs must be TensorBase instances.')

        if not all(isinstance(node, TensorBase) for node in nodes):
            raise TypeError('All nodes must be TensorBase instances.')
    
        self._inputs = inputs
        self._outputs = outputs
        for i in range(len(self._outputs)):
            if not self._outputs[i]._is_default_strides():
                self._outputs[i] = dt.basic._Copy(self._outputs[i])
                nodes.append(self._outputs[i])
        
        self._nodes = nodes
        self._constant_nodes: list[dt.core.Constant] = []
        self._variable_nodes: list[dt.core.Variable] = []
        self._compute_nodes: list[TensorBase] = []

        self._buffers: dict[dt.core.Buffer, cl.Buffer] = {}
        self._constants_loaded: bool = False
        self._kernels: OrderedDict[TensorBase, Callable[[], cl.Event | None]] = OrderedDict()

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
        target_tensors = [
            node
            for node in self._nodes
            if (node in self._inputs + self._outputs) or isinstance(node, dt.core.Update)
        ]

        nodes = topological_sort(target_tensors)
        self._nodes = [node for node in self._nodes if node in nodes]
    
    def _sort_nodes(self):
        dependencies: dict[TensorBase, set[TensorBase]] = {}
        update_dependencies: dict[TensorBase, set[TensorBase]] = {}
    
        for node in self._nodes:
            dependencies[node] = set(node.inputs())
            for inp in node.inputs():
                dependencies[node] |= update_dependencies[inp]
            update_dependencies[node] = dependencies[node].copy()

            if isinstance(node, dt.core.Update):
                update_dependencies[node._variable].add(node)
        
        node_indices: dict[TensorBase, int] = {}
        nodes: list[TensorBase] = []
        not_visited: set[TensorBase] = set(self._nodes)
        while not_visited:
            visited = set()
            for node in not_visited:
                if len(dependencies[node] & not_visited) == 0:
                    visited.add(node)
            
            not_visited -= visited

            key = lambda node: max([node_indices[inp] for inp in node.inputs()], default=0)
            visited_sorted = sorted(visited, key=key)
            for i, node in enumerate(visited_sorted, len(nodes)):
                nodes.append(node)
                node_indices[node] = i
            
        self._nodes = nodes
            

    def _filter_nodes(self):
        for node in self._nodes:
            if not self.device.is_support(node._dtype):
                raise NotSupportDtypeError(f'dtype {node.dtype} not supported on device {self.device}')
            
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

        # Вычисляем последний индекс использования каждого буфера
        last_usage: dict[dt.core.Buffer, int] = {}
        for idx, node in enumerate(self._nodes):
            buf = node._buffer
            last_usage[buf] = idx

        # Нумеруем буферы для идентификации
        unique_bufs = list({node._buffer for node in self._nodes})
        buf_id_map = {buf: i for i, buf in enumerate(unique_bufs)}

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'node_id', 'op_type', 'category', 'shape', 'dtype',
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

                input_ids = [self._nodes.index(inp) for inp in node.inputs()]

                buf = node._buffer
                buf_id = buf_id_map[buf]
                buf_bytes = buf.nbytes
                last_idx = last_usage[buf]

                writer.writerow([
                    idx, op_type, category, shape, dtype,
                    input_ids, buf_id, buf_bytes, last_idx
                ])

    def _get_free_buffer(self, allocated_cl_buffers: list[cl.Buffer], buffer: dt.core.Buffer) -> tuple[cl.Buffer, bool]:
        for cl_buffer in allocated_cl_buffers:
            if cl_buffer.size == buffer.nbytes:
                return (cl_buffer, True)
        return (cl.Buffer(self.device.context, cl.mem_flags.READ_WRITE, buffer.nbytes), False)
    
    def _allocate_buffers(self):
        need_buffers: list[dt.core.Buffer] = []
        for node in self._nodes:
            if node._buffer not in need_buffers:
                need_buffers.append(node._buffer)

        buffer_usage: dict[dt.core.Buffer, int] = {}
        for buffer in need_buffers:
            if buffer not in buffer_usage:
                buffer_usage[buffer] = 0
            
            for inp in buffer.inputs():
                buffer_usage[inp] += 1
        
        for node in self._variable_nodes + self._constant_nodes:
            buffer_usage[node._buffer] = -1
        
        free_cl_buffer: list[cl.Buffer] = []

        for buffer in need_buffers:
            cl_buffer, is_reused = self._get_free_buffer(free_cl_buffer, buffer)
            if is_reused:
                free_cl_buffer.remove(cl_buffer)

            self._buffers[buffer] = cl_buffer

            for inp in buffer.inputs():
                buffer_usage[inp] -= 1
                if buffer_usage[inp] == 0:
                    free_cl_buffer.append(self._buffers[inp])


    def _load_constants(self):
        if self._constants_loaded:
            return
        
        with dt.timestat.record('load_constants'):
            for const in self._constant_nodes:
                value = const.numpy()
                buffer = self._buffers[const._buffer]
                
                assert value.nbytes == buffer.size
                cl.enqueue_copy(self.device.queue, buffer, value)
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
                assert value.nbytes == buffer.size
                cl.enqueue_copy(self.device.queue, buffer, value)
            self.device.queue.finish()

    def _load_inputs(self, data: Sequence[np.ndarray | dt.core.Constant]):
        if len(self._inputs) != len(data):
            raise ValueError(f'Expected {len(self._inputs)} input(s), but got {len(data)}.')
        
        with dt.timestat.record('load_inputs'):
            for inp, array in zip(self._inputs, data):
                if inp.shape != array.shape:
                    raise ValueError(f'Shape mismatch for input {inp}: expected {inp.shape}, got {array.shape}')
                if isinstance(array, dt.core.Constant):
                    array = array._value
                
                array = array.astype(inp.dtype).copy()
                buffer = self._buffers[inp._buffer]
                cl.enqueue_copy(self.device.queue, buffer, array)
            self.device.queue.finish()
            
    def _get_results(self) -> list[dt.core.Constant]:
        result = []
        with dt.timestat.record('get_results'):
            for out in self._outputs:
                res = np.empty(out.shape, dtype=out.dtype)
                buffer = self._buffers[out._buffer]
                cl.enqueue_copy(self.device.queue, res, buffer)
                result.append(dt.constant(res))
            self.device.queue.finish()
            
        return result

    def _get_variable(self, var: dt.core.Variable) -> np.ndarray:
        if var not in self._variable_nodes:
            raise RuntimeError(f'Variable {var} is not part of this Function graph.')

        value = np.empty(var.shape, dtype=var.dtype)
        buffer = self._buffers[var._buffer]
        
        cl.enqueue_copy(self.device.queue, value, buffer)
        return value

    def _compile_kernels(self):
        if self._kernels:
            return
        with dt.timestat.record('compile_kernels'):
            for node in self._compute_nodes:
                input_buffers = [self._buffers[inp._buffer] for inp in node.inputs()]
                output_buffer = self._buffers[node._buffer]
                kernel = dt.compiler.compile_node(self.device, node, input_buffers, output_buffer)
                if kernel is not None:
                    self._kernels[node] = kernel

    def __call__(self, data: Sequence[np.ndarray | dt.core.Constant]) -> list[dt.core.Constant]:
        with dt.timestat.record('call'):
            self._load_variables()
            self._load_inputs(data)

            events: dict[TensorBase, cl.Event] = {}
            with dt.timestat.record('execute_kernels'):
                if dt.timestat.enabled():
                    for node, kernel in self._kernels.items():
                        s = str([(inp.shape, inp.dtype) for inp in node.inputs()])
                        s = f'{node}: {s}'
                        with dt.timestat.record(s):
                            kernel()
                            self.device.queue.finish()

                else:
                    for node, kernel in self._kernels.items():
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