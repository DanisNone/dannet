__kernel void onehot(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t idx = get_global_id(0);
    size_t shiftA = 0;

    for (int axis = 0; axis < ndimA; axis++)
        shiftA += ((idx / stridesAN[axis]) % shapeA[axis]) * stridesA[axis];
    
    dtypeA value = A[shiftA + offsetA];

    size_t shiftB = shiftA * shapeB[ndimB - 1];

    for (size_t i = 0; i < shapeB[ndimB - 1]; i++)
    {
        B[shiftB + i] = i == value? 1 : 0;
    }

}