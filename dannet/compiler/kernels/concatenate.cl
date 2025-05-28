__kernel void concatenate(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t shiftB = get_global_id(0);
    size_t shiftA = 0;

    for (int axis = 0; axis < ndimA; axis++)
        shiftA += ((shiftB / stridesAN[axis]) % shapeA[axis]) * stridesA[axis];

    B[shiftB + offsetB + concatenate_offset] = dt_convert_dtypeA_to_dtypeB(A[shiftA + offsetA]);   
}