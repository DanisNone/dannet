import dannet as dt
from dannet.topsort import topological_sort


def gradients(
    loss: dt.core.TensorBase,
    params: list[dt.core.TensorBase]
) -> list[dt.core.TensorBase]:
    if dt.is_eager():
        raise RuntimeError('gradients not work in eager mode')

    if not isinstance(loss, dt.core.TensorBase):
        raise TypeError('loss must be TensorBase instances.')
    
    for param in params:
        if not isinstance(param, dt.core.TensorBase):
            raise TypeError('All params must be TensorBase instances.')
    
    gradients: dict[dt.core.TensorBase, dt.core.TensorBase] = {
        loss: dt.ones_like(loss)}

    for node in topological_sort(loss):
        gradient = gradients[node]

        for inp, grad in zip(node.inputs(), node.compute_gradients(gradient)):
            if inp not in gradients:
                gradients[inp] = grad
            else:
                gradients[inp] += grad
    return [gradients.get(param, dt.zeros_like(param)) for param in params]
