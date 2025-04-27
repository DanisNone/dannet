from typing import Callable
import pyopencl as cl
import numpy as np
import dannet as dt
from dannet.topsort import topological_sort
from dannet import timestat


class compile:
    def __init__(self,
        inputs: list[dt.core.Placeholder],
        outputs: list[dt.core.TensorBase],
        nodes: list[dt.core.TensorBase]
    ):
        self.device = dt.current_device()

        if not all(isinstance(inp, dt.core.Placeholder) for inp in inputs):
            raise TypeError('All inputs must be Placeholder instances.')
        
        if not all(isinstance(out, dt.core.TensorBase) for out in outputs):
            raise TypeError('All outputs must be TensorBase instances.')

        if not all(isinstance(node, dt.core.TensorBase) for node in nodes):
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
        self._filtered_nodes: list[dt.core.TensorBase] = []

        self._filter_nodes()


        self._buffers: dict[dt.core.Buffer, cl.Buffer] = {}
        self._constants_loaded: bool = False
        self._kernels: list[Callable] = []
        
        self._allocate_buffers()
        self._compile_kernels()

        self._load_constants()

        with open('data.txt', 'w') as file:
            for node in nodes:
                idx = nodes.index(node)
                idxinp = [nodes.index(inp) for inp in node.inputs()]
                print(f'{idx}: {node}, {node.dtype}, {[inp.dtype for inp in node.inputs()]}', file=file)
        
    def _filter_nodes(self):
        target_tensors = [
            node
            for node in self._nodes
            if (node in self._outputs) or isinstance(node, dt.core.Update)
        ]
        nodes = topological_sort(target_tensors)
        self._nodes = [node for node in self._nodes if node in nodes]

        for node in self._nodes:
            if not self.device.is_support(node._dtype):
                raise TypeError(f'dtype {node.dtype} not supported on device {self.device}')
            
            if isinstance(node, dt.core.Constant):
                self._constant_nodes.append(node)
            elif isinstance(node, dt.core.Variable):
                self._variable_nodes.append(node)
            elif isinstance(node, dt.core.Placeholder):
                if node not in self._inputs:
                    raise ValueError('graph have unknown input')
            else:
                self._filtered_nodes.append(node)

    def _allocate_buffers(self):
        if self._buffers:
            return
        
        need_buffers: list[dt.core.Buffer] = []
        for node in self._nodes:
            if node._buffer not in need_buffers:
                need_buffers.append(node._buffer)
        
        for buffer in need_buffers:
            nbytes = buffer.nbytes
            cl_buffer = cl.Buffer(self.device.context, cl.mem_flags.READ_WRITE, nbytes)
            self._buffers[buffer] = cl_buffer

    def _load_constants(self):
        if self._constants_loaded:
            return
        
        with timestat.record('load_constants'):
            for const in self._constant_nodes:
                value = const.numpy()
                buffer = self._buffers[const._buffer]
                
                assert value.nbytes == buffer.size
                cl.enqueue_copy(self.device.queue, buffer, value)
            self._constants_loaded = True

    def _load_variables(self):
        with timestat.record('load_variables'):
            for var in self._variable_nodes:
                if var._used_by == self:
                    continue

                value = var.numpy()
                var._used_by = self

                buffer = self._buffers[var._buffer]
                assert value.nbytes == buffer.size
                cl.enqueue_copy(self.device.queue, buffer, value)
            self.device.queue.finish()

    def _load_inputs(self, data: list[np.ndarray]):
        if len(self._inputs) != len(data):
            raise ValueError(f'Expected {len(self._inputs)} input(s), but got {len(data)}.')
        
        with timestat.record('load_inputs'):
            for inp, array in zip(self._inputs, data):
                if inp.shape != array.shape:
                    raise ValueError(f'Shape mismatch for input {inp}: expected {inp.shape}, got {array.shape}')
                array = array.astype(inp.dtype).copy()
                buffer = self._buffers[inp._buffer]
                cl.enqueue_copy(self.device.queue, buffer, array)
            self.device.queue.finish()
            
    def _get_results(self) -> list[np.ndarray]:
        result = []
        with timestat.record('get_results'):
            for out in self._outputs:
                res = np.empty(out.shape, dtype=out.dtype)
                buffer = self._buffers[out._buffer]
                cl.enqueue_copy(self.device.queue, res, buffer)
                result.append(res)
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
        with timestat.record('compile_kernels'):
            for node in self._filtered_nodes:
                input_buffers = [self._buffers[inp._buffer] for inp in node.inputs()]
                output_buffer = self._buffers[node._buffer]
                kernel = dt.compiler.compile_node(self.device, node, input_buffers, output_buffer)
                self._kernels.append(kernel)

    def __call__(self, data: list[np.ndarray]) -> list[np.ndarray]:
        with timestat.record('call'):
            self._load_variables()
            self._load_inputs(data)

            with timestat.record('execute_kernels'):
                if dt.timestat.enabled():
                    for node, kernel in zip(self._filtered_nodes, self._kernels):
                        s = str([(inp.shape, inp.dtype) for inp in node.inputs()])
                        s = f"{node}: {s}"
                        with dt.timestat.record(s):
                            kernel()
                            self.device.queue.finish()

                else:
                    for node, kernel in zip(self._filtered_nodes, self._kernels):
                        kernel()
                        self.device.queue.finish()
            
            return self._get_results()
