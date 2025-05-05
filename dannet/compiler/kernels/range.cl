__kernel void range(__global dtypeA* A)
{
    size_t x = get_global_id(0);
    A[x] = x;
}