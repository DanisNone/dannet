__kernel void fill(
    __global dtypeA* A
)
{
    size_t shiftA = get_global_id(0) + offsetA;
    #ifdef zeros
        A[shiftA] = 0;
    #endif

    #ifdef ones
        A[shiftA] = 1;
    #endif
}