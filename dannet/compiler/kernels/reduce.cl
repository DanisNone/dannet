#ifdef general
__kernel void reduce(
    __global const dtypeA* A,
    __global dtypeB* B
)
{
    size_t shiftB = get_global_id(0);

    size_t shiftO = 0;

    for (int i = 0; i < ndimO; i++)
    {
        shiftO += ((shiftB / stridesON[i]) % shapeO[i]) * stridesO[i];
    }

    dtypeB res = init_value;
    for (size_t i = 0; i < sizeI; i++)
    {
        size_t shiftI = 0;
        for (int axis = 0; axis < ndimI; axis++)
        {
            shiftI += ((i / stridesIN[axis]) % shapeI[axis]) * stridesI[axis];
        }

        res = operation(res, A[shiftO + shiftI + offsetA]);
    }

    B[shiftB + offsetB] = final_operation(res, sizeI);
}
#endif