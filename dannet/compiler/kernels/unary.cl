#ifdef strided
__kernel void general(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t shiftB = get_global_id(0);
    size_t shiftA = 0;

    for (int axis = 0; axis < ndimB; axis++)
        shiftA += ((shiftB / stridesB[axis]) % shapeA[axis]) * stridesA[axis];

    B[shiftB + offsetB] = operation(A[shiftA + offsetA]);    
}
#endif

#ifdef full
__kernel void general(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t x = get_global_id(0);
    //size_t size = get_global_size(0);

    //for (; x < sizeB; x += size)
        B[x + offsetB] = operation(A[x + offsetA]);

}
#endif