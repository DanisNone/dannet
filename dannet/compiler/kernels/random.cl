#ifdef int_mode
__kernel void random(
    __global dt_uint64* A,
    dt_uint64 seed
)
{
    dt_uint64 index = get_global_id(0);
    
    dt_uint64 GOLDEN_RATIO = 0x9E3779B97F4A7C15;
    dt_uint64 x = seed + GOLDEN_RATIO * index;


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
    dt_uint64 seed
)
{
    dt_uint64 index = get_global_id(0);
    
    dt_uint64 GOLDEN_RATIO = 0x9E3779B97F4A7C15;
    dt_uint64 x = seed + GOLDEN_RATIO * index;


    // xorshift64 step
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    
    // normalize
    A[index] = dt_divide_dtypeA(
        dt_convert_uint64_to_dtypeA(x),
        dt_convert_uint64_to_dtypeA((dt_uint64)-1)
    );
}
#endif