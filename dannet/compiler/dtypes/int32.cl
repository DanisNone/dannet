#ifndef _INT32_CL_
#define _INT32_CL_


#include "dtypes/bool.cl"

typedef int dt_int32;

static inline dt_int32 dt_zero_int32() { return 0; }
static inline dt_int32 dt_one_int32() { return 1; }


dt_int32 dt_add_int32(dt_int32 x, dt_int32 y) {
    return x + y;
}

dt_int32 dt_subtract_int32(dt_int32 x, dt_int32 y) {
    return x - y;
}

dt_int32 dt_multiply_int32(dt_int32 x, dt_int32 y) {
    return x * y;
}

dt_int32 dt_divide_int32(dt_int32 x, dt_int32 y) {
    return x / y;
}

dt_int32 dt_floor_divide_int32(dt_int32 x, dt_int32 y) {
    dt_int32 q = x / y;
    dt_int32 r = x % y;

    return q - (dt_bool)((r != 0) && ((x ^ y) < 0));
}

dt_int32 dt_power_int32(dt_int32 x, dt_int32 y)
{
    if (y < 0)
        return x == 1;
    if (x == 0)
        return y == 0;

    dt_int32 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}

dt_int32 dt_bitwise_and_int32(dt_int32 x, dt_int32 y) {
    return x & y;
}

dt_int32 dt_bitwise_or_int32(dt_int32 x, dt_int32 y) {
    return x | y;
}

dt_int32 dt_bitwise_xor_int32(dt_int32 x, dt_int32 y) {
    return x ^ y;
}

dt_int32 dt_left_shift_int32(dt_int32 x, dt_int32 y) {
    return y >= (8*sizeof(y)) ? 0 : x << y;
}

dt_int32 dt_right_shift_int32(dt_int32 x, dt_int32 y) {
    return y >= (8*sizeof(y)) ? (-(x < 0)) : x >> y;
}

dt_int32 dt_negative_int32(dt_int32 x) {
    return -x;
}

dt_int32 dt_positive_int32(dt_int32 x) {
    return x;
}


dt_int32 dt_bitwise_not_int32(dt_int32 x) {
    return ~x;
}

dt_int32 dt_square_int32(dt_int32 x) {
    return x * x;
}


dt_int32 dt_abs_int32(dt_int32 x) {
    return abs(x);
}

dt_int32 dt_sign_int32(dt_int32 x) {
    return (x > 0) - (x < 0);
}


dt_bool dt_equal_int32(dt_int32 x, dt_int32 y) {
    return x == y;
}

dt_bool dt_not_equal_int32(dt_int32 x, dt_int32 y) {
    return x != y;
}

dt_bool dt_greater_int32(dt_int32 x, dt_int32 y) {
    return x > y;
}

dt_bool dt_greater_equal_int32(dt_int32 x, dt_int32 y) {
    return x >= y;
}

dt_bool dt_less_int32(dt_int32 x, dt_int32 y) {
    return x < y;
}

dt_bool dt_less_equal_int32(dt_int32 x, dt_int32 y) {
    return x <= y;
}

dt_bool dt_logical_and_int32(dt_int32 x, dt_int32 y) {
    return x && y;
}

dt_bool dt_logical_or_int32(dt_int32 x, dt_int32 y) {
    return x || y;
}

dt_bool dt_logical_xor_int32(dt_int32 x, dt_int32 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_int32(dt_int32 x) {
    return x == 0;
}

dt_int32 dt_min_int32(dt_int32 x, dt_int32 y) {
    return min(x, y);
}

dt_int32 dt_max_int32(dt_int32 x, dt_int32 y) {
    return max(x, y);
}

dt_int32 dt_mad_int32(dt_int32 x, dt_int32 y, dt_int32 z) {
    return x * y + z;
}

#endif