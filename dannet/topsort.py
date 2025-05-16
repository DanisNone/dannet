import dannet as dt


def topological_sort(
    tensors: list[dt.core.TensorBase] | dt.core.TensorBase
) -> list[dt.core.TensorBase]:
    result = []
    if isinstance(tensors, dt.core.TensorBase):
        tensors = [tensors]
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
    return result[::-1]
