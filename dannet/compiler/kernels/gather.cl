__kernel void gather(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeA* C
)
{
    size_t shiftC = get_global_id(0);

    size_t shiftA = offsetA;
    size_t shiftB = offsetB;

    for (int axis = 0; axis < ndimC; axis++) {
        size_t coord = (shiftC / stridesC[axis]) % shapeC[axis];
        if (axis < ndimB)
            shiftB += coord * stridesBN[axis];
        else
            shiftA += coord * stridesA[axis - ndimB + 1];
    }

    int ix = B[shiftB];
    
    if (ix < 0) ix += shapeA[0];

    shiftA += ix * stridesA[0];

    C[shiftC + offsetC] = A[shiftA];
}