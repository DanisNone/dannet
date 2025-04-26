#ifdef strided
__kernel void general(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global const dtypeC* C,
    __global dtypeD* D
)
{
    size_t shiftD = get_global_id(0);
    size_t shiftA = 0;
    size_t shiftB = 0;
    size_t shiftC = 0;

    for (int axis = 0; axis < ndimD; axis++)
    {
        size_t coord = shiftD / stridesD[axis];
        shiftA += (coord % shapeA[axis]) * stridesA[axis];
        shiftB += (coord % shapeB[axis]) * stridesB[axis];
        shiftC += (coord % shapeC[axis]) * stridesC[axis];
    }

    D[shiftD + offsetD] = operation(A[shiftA + offsetA], B[shiftB + offsetB], C[shiftC + offsetC]);    
}
#endif