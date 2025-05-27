import math

import opt_einsum
import opt_einsum.typing
import dannet as dt


def _einsum_one_arg(einsum_str, operands: list):
    einsum_inputs, einsum_output = einsum_str.split('->')
    inputs = einsum_inputs.split(',')
    if len(inputs) != len(operands) != 1:
        raise ValueError('Expected one input subscripts and one operands')

    input, = inputs
    if len(input) != len(set(input)):
        # TODO: implement dt.diagonal
        raise ValueError()

    sum_axes = [i for i, c in enumerate(input) if c not in einsum_output]
    input = ''.join(c for c in input if c in einsum_output)

    assert len(input) == len(einsum_output)

    perm = [input.index(c) for c in einsum_output]

    sum_ = dt.core._node_prepare(
        dt.reduce._DefaultDtypeSum(operands[0], sum_axes)
    )
    return dt.transpose(sum_, perm)


def _einsum(einsum_str: str, operands: list):
    einsum_inputs, einsum_output = einsum_str.split('->')
    if ',' not in einsum_inputs:
        return _einsum_one_arg(einsum_str, operands)

    inputs = einsum_inputs.split(',')

    if len(inputs) != len(operands) != 2:
        raise ValueError('Expected two input subscripts and two operands')

    if any(len(inp) != len(set(inp)) for inp in inputs):
        # TODO: implement dt.diagonal
        raise ValueError

    ids = set(einsum_str) - set(',->')
    id2num = {c: i for i, c in enumerate(sorted(ids))}

    Aids = tuple(id2num[c] for c in inputs[0])
    Bids = tuple(id2num[c] for c in inputs[1])
    Oids = tuple(id2num[c] for c in einsum_output)

    setA = set(Aids)
    setB = set(Bids)
    setO = set(Oids)

    batch = sorted(setA & setB & setO)
    left = sorted(setA & setO - setB)
    right = sorted(setB & setO - setA)
    remove = sorted(setA & setB - setO)

    A, B = operands

    A_order = batch + left + remove
    B_order = batch + remove + right

    A_t = dt.transpose(A, [Aids.index(idx) for idx in A_order])
    B_t = dt.transpose(B, [Bids.index(idx) for idx in B_order])

    dims = {id2num[c]: dim for c, dim in zip(inputs[0], A.shape)}
    dims.update({id2num[c]: dim for c, dim in zip(inputs[1], B.shape)})

    batch_shape = [dims[i] for i in batch]
    M = math.prod(dims[i] for i in left)
    K = math.prod(dims[i] for i in remove)
    N = math.prod(dims[i] for i in right)

    A_rs = dt.reshape(A_t, (-1, M, K))
    B_rs = dt.reshape(B_t, (-1, K, N))

    O_rs = dt.matmul(A_rs, B_rs)

    out_shape = [
        *batch_shape,
        *[dims[i] for i in left],
        *[dims[i] for i in right]
    ]
    out = dt.reshape(O_rs, out_shape)

    current_order = batch + left + right
    perm = [current_order.index(idx) for idx in Oids]
    out = dt.transpose(out, perm)
    return out


def einsum(
    subscripts: str,
    *operands: dt.typing.TensorLike,
    optimize: opt_einsum.typing.OptimizeKind = 'auto'
):
    if optimize is False:
        raise ValueError('dannet.einsum not support optimize=False')

    operands = tuple(dt.convert_to_tensor(op) for op in operands)
    contraction_list: opt_einsum.typing.ContractionListType
    operands_list: list
    operands_list, contraction_list = opt_einsum.contract_path(
        subscripts, *operands, optimize=optimize, einsum_call=True
    )  # type: ignore

    for contraction in contraction_list:
        inds, idx_rm, einsum_str, remaining, tensordot = contraction
        tmp_operands = [operands_list.pop(x) for x in inds]

        if tensordot:
            input_str, results_index = einsum_str.split('->')
            input_left, input_right = input_str.split(',')

            tensor_result = input_left + input_right
            for s in idx_rm:
                tensor_result = tensor_result.replace(s, '')

            left_pos, right_pos = [], []
            for s in sorted(idx_rm):
                left_pos.append(input_left.find(s))
                right_pos.append(input_right.find(s))

            result = dt.tensordot(
                *tmp_operands, axes=(tuple(left_pos), tuple(right_pos))
            )

            if tensor_result != results_index:
                transpose = tuple(map(tensor_result.index, results_index))
                result = dt.transpose(result, transpose)

        else:
            try:
                result = _einsum(einsum_str, tmp_operands)
            except ValueError:
                raise NotImplementedError(
                    f'einsum args: {einsum_str}, '
                    f'{[op.shape for op in tmp_operands]}'
                )

        operands_list.append(result)
        del tmp_operands, result

    return dt.convert_to_tensor(operands_list[0])
