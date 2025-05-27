__kernel void upsamplezeros(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t shiftB = get_global_id(0);
    size_t shiftA = 0;
    bool is_zero = false;

    for (int axis = 0; axis < ndimA; axis++)
    {
        size_t coord = (shiftB / stridesB[axis]) % shapeB[axis];

        if (coord % upsample_size[axis] == 0)
            shiftA += (coord / upsample_size[axis]) * stridesA[axis];
        else
        {
            is_zero = true;
            break;
        }
    }

    if (is_zero)
        B[shiftB] = dt_zero_dtypeB();
    else
        B[shiftB] = dt_convert_dtypeA_to_dtypeB(A[shiftA + offsetA]);
}