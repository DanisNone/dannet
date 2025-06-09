#ifndef _UINT64_CL_
#define _UINT64_CL_

#include "dtypes/bool.cl"

typedef ulong dt_uint64;

static inline dt_uint64 dt_zero_uint64() { return 0; }
static inline dt_uint64 dt_one_uint64() { return 1; }


dt_uint64 dt_add_uint64(dt_uint64 x, dt_uint64 y) {
    return x + y;
}

dt_uint64 dt_subtract_uint64(dt_uint64 x, dt_uint64 y) {
    return x - y;
}

dt_uint64 dt_multiply_uint64(dt_uint64 x, dt_uint64 y) {
    return x * y;
}

dt_uint64 dt_divide_uint64(dt_uint64 x, dt_uint64 y) {
    return x / y;
}


dt_uint64 dt_floor_divide_uint64(dt_uint64 x, dt_uint64 y) {
    return x / y;
}

dt_uint64 dt_power_uint64(dt_uint64 x, dt_uint64 y)
{
    if (x == 0)
        return y == 0;

    dt_uint64 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}


dt_uint64 dt_bitwise_and_uint64(dt_uint64 x, dt_uint64 y) {
    return x & y;
}

dt_uint64 dt_bitwise_or_uint64(dt_uint64 x, dt_uint64 y) {
    return x | y;
}

dt_uint64 dt_bitwise_xor_uint64(dt_uint64 x, dt_uint64 y) {
    return x ^ y;
}

dt_uint64 dt_left_shift_uint64(dt_uint64 x, dt_uint64 y) {
    return y >= (sizeof(y)*8) ? 0 : x << y;
}

dt_uint64 dt_right_shift_uint64(dt_uint64 x, dt_uint64 y) {
    return y >= (sizeof(y)*8) ? 0 : x >> y;
}

dt_uint64 dt_negative_uint64(dt_uint64 x) {
    return -x;
}

dt_uint64 dt_positive_uint64(dt_uint64 x) {
    return x;
}

dt_uint64 dt_bitwise_not_uint64(dt_uint64 x) {
    return ~x;
}

dt_uint64 dt_square_uint64(dt_uint64 x) {
    return x * x;
}

dt_uint64 dt_abs_uint64(dt_uint64 x) {
    return x;
}

dt_uint64 dt_sign_uint64(dt_uint64 x) {
    return x > 0;
}

dt_bool dt_equal_uint64(dt_uint64 x, dt_uint64 y) {
    return x == y;
}

dt_bool dt_not_equal_uint64(dt_uint64 x, dt_uint64 y) {
    return x != y;
}

dt_bool dt_greater_uint64(dt_uint64 x, dt_uint64 y) {
    return x > y;
}

dt_bool dt_greater_equal_uint64(dt_uint64 x, dt_uint64 y) {
    return x >= y;
}

dt_bool dt_less_uint64(dt_uint64 x, dt_uint64 y) {
    return x < y;
}

dt_bool dt_less_equal_uint64(dt_uint64 x, dt_uint64 y) {
    return x <= y;
}


dt_bool dt_logical_and_uint64(dt_uint64 x, dt_uint64 y) {
    return x && y;
}

dt_bool dt_logical_or_uint64(dt_uint64 x, dt_uint64 y) {
    return x || y;
}

dt_bool dt_logical_xor_uint64(dt_uint64 x, dt_uint64 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_uint64(dt_uint64 x) {
    return x == 0;
}

dt_uint64 dt_min_uint64(dt_uint64 x, dt_uint64 y) {
    return min(x, y);
}

dt_uint64 dt_max_uint64(dt_uint64 x, dt_uint64 y) {
    return max(x, y);
}

dt_uint64 dt_mad_uint64(dt_uint64 x, dt_uint64 y, dt_uint64 z) {
    return x * y + z;
}

#endif