#ifdef full
__kernel void reduce(__global const dtypeA *A, __global dtypeB *B)
{
    // global_size == 1
    // global_id == 0

    dtypeB index = dt_zero_dtypeB();
    dtypeA value = A[offsetA];

    for (size_t i = 0; i < sizeA; i++)
    {
        size_t shiftA = 0;
        for (int axis = 0; axis < ndimA; axis++)
            shiftA += ((i / stridesAN[axis]) % shapeA[axis]) * stridesA[axis];

        dtypeA current = A[shiftA + offsetA];

        if (condition(value, current))
        {
            index = i;
            value = current;
        }
    }

    B[offsetB] = index;
}
#endif

#ifdef by_axis
__kernel void reduce(__global const dtypeA *A, __global dtypeB *B)
{
    size_t shiftB = get_global_id(0);
    size_t shiftBR = (shiftB / sizeRight) * sizeRight * shapeA[skeep_axis] + (shiftB % sizeRight);
    size_t shiftO = 0;

    for (int axis = 0; axis < ndimA; axis++)
    {   
        if (axis != skeep_axis)
            shiftO += ((shiftBR / stridesAN[axis]) % shapeA[axis]) * stridesA[axis];
    }

    dtypeB index = dt_zero_dtypeB();
    dtypeA value = A[shiftO + offsetA];
    for (size_t i = 0; i < shapeA[skeep_axis]; i++)
    {
        
        size_t shiftI = i * stridesA[skeep_axis];
        dtypeA current = A[shiftO + shiftI + offsetA];

        if (condition(value, current))
        {
            index = i;
            value = current;
        }
    }

    B[shiftB + offsetB] = index;
}
#endif