#ifdef zero
__kernel void padding(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t shiftB = get_global_id(0);
    size_t shiftA = 0;

    bool is_zero = false;

    for (int i = 0; i < ndimB; i++)
    {
        size_t coord = (shiftB / stridesB[i]) % shapeB[i];
        coord -= pad_left[i];
        if (coord >= shapeA[i])
        {
            is_zero = true;
            break;
        }

        shiftA += coord * stridesA[i];
    }

    if (is_zero)
        B[shiftB + offsetB] = dt_zero_dtypeB();
    else
        B[shiftB + offsetB] = dt_convert_dtypeA_to_dtypeB(A[shiftA + offsetA]);
}
#endif