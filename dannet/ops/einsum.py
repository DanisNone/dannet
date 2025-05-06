import operator
from typing import Sequence
import dannet as dt


einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
einsum_symbols_set = set(einsum_symbols)


def _parse_einsum_input(operands: Sequence):
    if len(operands) == 0:
        raise ValueError('No input operands')

    if isinstance(operands[0], str):
        subscripts: str = operands[0].replace(' ', '')
        operands = [v for v in operands[1:]]

        # Ensure all characters are valid
        for s in subscripts:
            if s in '.,->':
                continue
            if s not in einsum_symbols:
                raise ValueError('Character %s is not a valid symbol.' % s)

    else:
        tmp_operands = list(operands)
        operand_list = []
        subscript_list = []
        for p in range(len(operands) // 2):
            operand_list.append(tmp_operands.pop(0))
            subscript_list.append(tmp_operands.pop(0))

        output_list = tmp_operands[-1] if len(tmp_operands) else None
        operands = [dt.convert_to_tensor(v) for v in operand_list]
        subscripts = str()
        last = len(subscript_list) - 1
        for num, sub in enumerate(subscript_list):
            for s in sub:
                if s is Ellipsis:
                    subscripts += '...'
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            'For this input type lists must contain '
                            'either int or Ellipsis'
                        ) from e
                    subscripts += einsum_symbols[s]
            if num != last:
                subscripts += ','

        if output_list is not None:
            subscripts += '->'
            for s in output_list:
                if s is Ellipsis:
                    subscripts += '...'
                else:
                    try:
                        s = operator.index(s)
                    except TypeError as e:
                        raise TypeError(
                            'For this input type lists must contain '
                            'either int or Ellipsis'
                        ) from e
                    subscripts += einsum_symbols[s]
    
    # Check for proper '->'
    if ('-' in subscripts) or ('>' in subscripts):
        invalid = (subscripts.count('-') > 1) or (subscripts.count('>') > 1)
        if invalid or (subscripts.count('->') != 1):
            raise ValueError('Subscripts can only contain one \'->\'.')

    # Parse ellipses
    if '.' in subscripts:
        used = subscripts.replace('.', '').replace(',', '').replace('->', '')
        unused = list(einsum_symbols_set - set(used))
        ellipse_inds = ''.join(unused)
        longest = 0

        if '->' in subscripts:
            input_tmp, output_sub = subscripts.split('->')
            split_subscripts: list[str] = input_tmp.split(',')
        else:
            split_subscripts, output_sub = subscripts.split(','), None

        for num, sub in enumerate(split_subscripts):
            if '.' in sub:
                if (sub.count('.') != 3) or (sub.count('...') != 1):
                    raise ValueError('Invalid Ellipses.')

                # Take into account numerical values
                if operands[num].shape == ():
                    ellipse_count = 0
                else:
                    ellipse_count = max(operands[num].ndim, 1)
                    ellipse_count -= (len(sub) - 3)

                if ellipse_count > longest:
                    longest = ellipse_count

                if ellipse_count < 0:
                    raise ValueError('Ellipses lengths do not match.')
                elif ellipse_count == 0:
                    split_subscripts[num] = sub.replace('...', '')
                else:
                    rep_inds = ellipse_inds[-ellipse_count:]
                    split_subscripts[num] = sub.replace('...', rep_inds)

        subscripts = ','.join(split_subscripts)
        if longest == 0:
            out_ellipse = ''
        else:
            out_ellipse = ellipse_inds[-longest:]

        if output_sub is not None:
            subscripts += '->' + output_sub.replace('...', out_ellipse)
        else:
            # Special care for outputless ellipses
            output_subscript = ''
            tmp_subscripts = subscripts.replace(',', '')
            for s in sorted(set(tmp_subscripts)):
                if s not in (einsum_symbols):
                    raise ValueError('Character %s is not a valid symbol.' % s)
                if tmp_subscripts.count(s) == 1:
                    output_subscript += s
            normal_inds = ''.join(sorted(set(output_subscript) -
                                         set(out_ellipse)))

            subscripts += '->' + out_ellipse + normal_inds

    # Build output string if does not exist
    if '->' in subscripts:
        input_subscripts, output_subscript = subscripts.split('->')
    else:
        input_subscripts = subscripts
        # Build output subscripts
        tmp_subscripts = subscripts.replace(',', '')
        output_subscript = ''
        for s in sorted(set(tmp_subscripts)):
            if s not in einsum_symbols:
                raise ValueError('Character %s is not a valid symbol.' % s)
            if tmp_subscripts.count(s) == 1:
                output_subscript += s

    # Make sure output subscripts are in the input
    for char in output_subscript:
        if output_subscript.count(char) != 1:
            raise ValueError('Output character %s appeared more than once in '
                             'the output.' % char)
        if char not in input_subscripts:
            raise ValueError('Output character %s did not appear in the input'
                             % char)

    # Make sure number operands is equivalent to the number of terms
    if len(input_subscripts.split(',')) != len(operands):
        raise ValueError('Number of einsum subscripts must be equal to the '
                         'number of operands.')

    return (input_subscripts, output_subscript, operands)

def _find_contraction(positions, input_sets, output_set):
    idx_contract = set()
    idx_remain = output_set.copy()
    remaining = []
    for ind, value in enumerate(input_sets):
        if ind in positions:
            idx_contract |= value
        else:
            remaining.append(value)
            idx_remain |= value

    new_result = idx_remain & idx_contract
    idx_removed = (idx_contract - new_result)
    remaining.append(new_result)

    return (new_result, remaining, idx_removed, idx_contract)

def _can_dot(inputs, result, idx_removed):
    # All `dot` calls remove indices
    if len(idx_removed) == 0:
        return False

    # tensordot can only handle two operands
    if len(inputs) != 2:
        return False

    input_left, input_right = inputs

    for c in set(input_left + input_right):
        # can't deal with repeated indices on same input or more than 2 total
        nl, nr = input_left.count(c), input_right.count(c)
        if (nl > 1) or (nr > 1) or (nl + nr > 2):
            return False

        # can't do implicit summation or dimension collapse e.g.
        #     'ab,bc->c' (implicitly sum over 'a')
        #     'ab,ca->ca' (take diagonal of 'a')
        if nl + nr - 1 == int(c in result):
            return False

    # Build a few temporaries
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)

    # Handle inner products

    # DDOT with aligned data
    if input_left == input_right:
        return True

    # DDOT without aligned data (better to use einsum)
    if set_left == set_right:
        return False

    # Handle the 4 possible (aligned) GEMV or GEMM cases

    # GEMM or GEMV no transpose
    if input_left[-rs:] == input_right[:rs]:
        return True

    # GEMM or GEMV transpose both
    if input_left[:rs] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose right
    if input_left[-rs:] == input_right[-rs:]:
        return True

    # GEMM or GEMV transpose left
    if input_left[:rs] == input_right[:rs]:
        return True

    # Einsum is faster than GEMV if we have to copy data
    if not keep_left or not keep_right:
        return False

    # We are a matrix-matrix product, but we need to copy data
    return True

def einsum_path(*operands):
    input_subscripts, output_subscript, operands = (
        _parse_einsum_input(operands)
    )

    # Build a few useful list and sets
    input_list = input_subscripts.split(',')
    input_sets = [set(x) for x in input_list]
    output_set = set(output_subscript)

    # Get length of each unique dimension and ensure all dimensions are correct
    dimension_dict = {}
    for tnum, term in enumerate(input_list):
        sh = operands[tnum].shape
        if len(sh) != len(term):
            raise ValueError('Einstein sum subscript %s does not contain the '
                             'correct number of indices for operand %d.'
                             % (input_subscripts[tnum], tnum))
        for cnum, char in enumerate(term):
            dim = sh[cnum]
            if char in dimension_dict.keys():
                # For broadcasting cases we always want the largest dim size
                if dimension_dict[char] == 1:
                    dimension_dict[char] = dim
                elif dim not in (1, dimension_dict[char]):
                    raise ValueError('Size of label \'%s\' for operand %d (%d) '
                                     'does not match previous terms (%d).'
                                     % (char, tnum, dimension_dict[char], dim))
            else:
                dimension_dict[char] = dim
    for tnum, term in enumerate(input_list):
        shape = tuple(dimension_dict[char] for char in term)
        operands[tnum] = dt.broadcast_to(operands[tnum], shape)

    path = [tuple(range(len(input_list)))]
    contraction_list = []
    for cnum, contract_inds in enumerate(path):
        contract_inds = tuple(sorted(contract_inds, reverse=True))

        contract = _find_contraction(contract_inds, input_sets, output_set)
        out_inds, input_sets, idx_removed, idx_contract = contract

        bcast = set()
        tmp_inputs = []
        for x in contract_inds:
            tmp_inputs.append(input_list.pop(x))


        if not len(idx_removed & bcast):
            do_tensordot = _can_dot(tmp_inputs, out_inds, idx_removed)
        else:
            do_tensordot = False

        # Last contraction
        if (cnum - len(path)) == -1:
            idx_result = output_subscript
        else:
            sort_result = [(dimension_dict[ind], ind) for ind in out_inds]
            idx_result = ''.join([x[1] for x in sorted(sort_result)])

        input_list.append(idx_result)
        einsum_str = ','.join(tmp_inputs) + '->' + idx_result

        contraction = (
            contract_inds, idx_removed, einsum_str, input_list[:], do_tensordot
        )
        contraction_list.append(contraction)


    if len(input_list) != 1:
        raise RuntimeError(
            'Invalid einsum_path is specified: {} more operands has to be '
            'contracted.'.format(len(input_list) - 1))
    return (operands, contraction_list)

def einsum(*operands):
    operands, contraction_list = einsum_path(*operands)

    for num, contraction in enumerate(contraction_list):
        inds, idx_rm, einsum_str, remaining, tensordot = contraction
        tmp_operands = [operands.pop(x) for x in inds]

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
                if sorted(tensor_result) != sorted(results_index):
                    einsum_str = tensor_result + '->' + results_index
                    raise NotImplementedError(f'einsum args: {einsum_str}')

                perm = [tensor_result.index(c) for c in results_index]
                result = dt.transpose(result, perm)

        else:
            raise NotImplementedError(f'einsum args: {einsum_str}, {tmp_operands}')

        operands.append(result)
        del tmp_operands, result

    return dt.convert_to_tensor(operands[0])
