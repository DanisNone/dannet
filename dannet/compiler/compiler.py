from typing import Sequence

from dannet.device import Device, DeviceEvent
from dannet.graph_collections import GDict, GList, GSet
from dannet.lib import core
from dannet.lib.core import SymbolicTensor
from dannet.compiler.register import compile_node, kernel_func_type


def topological_sort(
    tensors: Sequence[SymbolicTensor] | SymbolicTensor
) -> list[SymbolicTensor]:
    if isinstance(tensors, SymbolicTensor):
        tensors = [tensors]

    result: GList[SymbolicTensor] = GList()
    visited: GSet[SymbolicTensor] = GSet()
    stack = list(reversed(tensors))

    while stack:
        x = stack.pop()
        if x in visited:
            continue

        inputs = [inp for inp in x.inputs() if inp not in visited]

        if inputs:
            stack.append(x)
            stack.extend(inputs)
        else:
            result.append(x)
            visited.add(x)
    return result[::-1].tolist()


class compile:
    def __init__(
        self, device: Device,
        inputs: list[core.Placeholder],
        outputs: list[SymbolicTensor]
    ):
        if not all(isinstance(inp, core.Placeholder) for inp in inputs):
            raise TypeError('All inputs must be Placeholder instances.')

        if not all(isinstance(out, SymbolicTensor) for out in outputs):
            raise TypeError('All outputs must be SymbolicBase instances.')

        if len(GSet(inputs)) != len(inputs):
            raise ValueError
        self.device = device
        self.inputs = GList(inputs)
        self.outputs = GList(outputs)

        self.nodes = topological_sort(self.outputs)[::-1]
        self.input_buffers = GList(inp.buffer for inp in self.inputs)
        self.output_buffers = GList(out.buffer for out in self.outputs)
        self.buffers: GDict[core.SymbolicBuffer, core.ConcreteBuffer] = GDict()
        self.kernels: list[tuple[SymbolicTensor, kernel_func_type]] = []

        self.allocate_buffers()
        self.compile_kernels()

    def allocate_buffers(self) -> None:
        assert not self.buffers
        buffers = GSet(node.buffer for node in self.nodes)

        for buffer in buffers:
            if buffer._tensor.buffer in self.input_buffers:
                continue
            if buffer._tensor.buffer in self.output_buffers:
                continue
            if isinstance(buffer._tensor, core.Constant):
                concrete = buffer._tensor._concrete_tensor.buffer
            else:
                concrete = core.ConcreteBuffer(self.device, buffer.nbytes)
            self.buffers[buffer] = concrete

    def compile_kernels(self) -> None:
        assert not self.kernels

        for node in self.nodes:
            self.kernels.append(
                (node, compile_node(self.device, node))
            )

    def __call__(self, values: list[core.ConcreteTensor]) -> list[core.ConcreteTensor]:
        if len(values) != len(self.inputs):
            raise
        events: GDict[SymbolicTensor, DeviceEvent] = GDict()

        buffers = self.buffers.copy()
        for placeholder, value in zip(self.inputs, values):
            if (
                placeholder.shape != value.shape or
                placeholder.strides != value.strides or
                placeholder.buffer_offset != value.buffer_offset or
                placeholder.dtype != value.dtype
            ):
                raise

            assert placeholder.buffer not in buffers, buffers
            buffers[placeholder.buffer] = value.buffer
            if value._event:
                events[placeholder] = value._event

        for out in self.outputs:
            if out.buffer in buffers:
                continue
            buffers[out.buffer] = core.ConcreteBuffer(
                self.device, out.buffer.nbytes)

        for node, kernel in self.kernels:
            inputs = [core.to_concrete(inp, buffers[inp.buffer])
                      for inp in node.inputs()]
            output = core.to_concrete(node, buffers[node.buffer])

            for inp in node.inputs():
                if inp in events:
                    events.pop(inp).wait()

            event = kernel(inputs, output)
            if event:
                events[node] = event

        result = [
            core.to_concrete(out, buffers[out.buffer], events.get(out))
            for out in self.outputs
        ]
        return result
