#ifdef strided
__kernel void binary(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC* C
)
{
    size_t shiftC = get_global_id(0);
    size_t shiftA = 0;
    size_t shiftB = 0;

    for (int axis = 0; axis < ndimC; axis++)
    {
        size_t coord = shiftC / stridesC[axis];
        shiftA += (coord % shapeA[axis]) * stridesA[axis];
        shiftB += (coord % shapeB[axis]) * stridesB[axis];
    }

    C[shiftC + offsetC] = operation(A[shiftA + offsetA], B[shiftB + offsetB]);    
}
#endif

#ifdef full
__kernel void binary(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC* C
)
{
    size_t x = get_global_id(0);
    C[x + offsetC] = operation(A[x + offsetA], B[x + offsetB]);    
}
#endif