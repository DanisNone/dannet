#ifndef _UINT32_CL_
#define _UINT32_CL_

#include "dtypes/bool.cl"

typedef uint dt_uint32;

static inline dt_uint32 dt_zero_uint32() { return 0; }
static inline dt_uint32 dt_one_uint32() { return 1; }


dt_uint32 dt_add_uint32(dt_uint32 x, dt_uint32 y) {
    return x + y;
}

dt_uint32 dt_subtract_uint32(dt_uint32 x, dt_uint32 y) {
    return x - y;
}

dt_uint32 dt_multiply_uint32(dt_uint32 x, dt_uint32 y) {
    return x * y;
}

dt_uint32 dt_divide_uint32(dt_uint32 x, dt_uint32 y) {
    return x / y;
}


dt_uint32 dt_floor_divide_uint32(dt_uint32 x, dt_uint32 y) {
    return x / y;
}

dt_uint32 dt_power_uint32(dt_uint32 x, dt_uint32 y)
{
    if (x == 0)
        return y == 0;

    dt_uint32 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}


dt_uint32 dt_bitwise_and_uint32(dt_uint32 x, dt_uint32 y) {
    return x & y;
}

dt_uint32 dt_bitwise_or_uint32(dt_uint32 x, dt_uint32 y) {
    return x | y;
}

dt_uint32 dt_bitwise_xor_uint32(dt_uint32 x, dt_uint32 y) {
    return x ^ y;
}

dt_uint32 dt_left_shift_uint32(dt_uint32 x, dt_uint32 y) {
    return y >= (sizeof(y)*8) ? 0 : x << y;
}

dt_uint32 dt_right_shift_uint32(dt_uint32 x, dt_uint32 y) {
    return y >= (sizeof(y)*8) ? 0 : x >> y;
}

dt_uint32 dt_negative_uint32(dt_uint32 x) {
    return -x;
}

dt_uint32 dt_bitwise_not_uint32(dt_uint32 x) {
    return ~x;
}

dt_uint32 dt_square_uint32(dt_uint32 x) {
    return x * x;
}

dt_uint32 dt_abs_uint32(dt_uint32 x) {
    return x;
}

dt_uint32 dt_sign_uint32(dt_uint32 x) {
    return x > 0;
}

dt_bool dt_equal_uint32(dt_uint32 x, dt_uint32 y) {
    return x == y;
}

dt_bool dt_not_equal_uint32(dt_uint32 x, dt_uint32 y) {
    return x != y;
}

dt_bool dt_greater_uint32(dt_uint32 x, dt_uint32 y) {
    return x > y;
}

dt_bool dt_greater_equal_uint32(dt_uint32 x, dt_uint32 y) {
    return x >= y;
}

dt_bool dt_less_uint32(dt_uint32 x, dt_uint32 y) {
    return x < y;
}

dt_bool dt_less_equal_uint32(dt_uint32 x, dt_uint32 y) {
    return x <= y;
}


dt_bool dt_logical_and_uint32(dt_uint32 x, dt_uint32 y) {
    return x && y;
}

dt_bool dt_logical_or_uint32(dt_uint32 x, dt_uint32 y) {
    return x || y;
}

dt_bool dt_logical_xor_uint32(dt_uint32 x, dt_uint32 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_uint32(dt_uint32 x) {
    return x == 0;
}

dt_uint32 dt_min_uint32(dt_uint32 x, dt_uint32 y) {
    return min(x, y);
}

dt_uint32 dt_max_uint32(dt_uint32 x, dt_uint32 y) {
    return max(x, y);
}

dt_uint32 dt_mad_uint32(dt_uint32 x, dt_uint32 y, dt_uint32 z) {
    return x * y + z;
}

#endif