from typing import Sequence
import dannet as dt

from dannet.graph_collections import GList


def topological_sort(
    tensors: Sequence[dt.core.TensorBase] | dt.core.TensorBase
) -> list[dt.core.TensorBase]:
    result: GList[dt.core.TensorBase] = GList()

    if isinstance(tensors, dt.core.TensorBase):
        tensors = [tensors]
    tensors = list(tensors)
    stack = tensors[::-1].copy()

    while stack:
        x = stack.pop()
        if x in result:
            continue

        inputs = [inp for inp in x.inputs() if inp not in result]

        if inputs:
            stack.append(x)
            stack.extend(inputs)
        else:
            result.append(x)
    return result[::-1].tolist()
