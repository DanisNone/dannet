#ifdef int_mode
__kernel void random(
    __global ulong* A,
    ulong seed
)
{
    ulong index = get_global_id(0);
    
    ulong GOLDEN_RATIO = 0x9E3779B97F4A7C15;
    ulong x = seed + GOLDEN_RATIO * index;


    // xorshift64 step
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    
    A[index] = x;
}
#endif


#ifdef float_mode
__kernel void random_float(
    __global dtypeA* A,
    ulong seed
)
{
    ulong index = get_global_id(0);
    
    ulong GOLDEN_RATIO = 0x9E3779B97F4A7C15;
    ulong x = seed + GOLDEN_RATIO * index;


    // xorshift64 step
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    
    // normalize
    A[index] = x / (dtypeA)((ulong)-1);
}
#endif