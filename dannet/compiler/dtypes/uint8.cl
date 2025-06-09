#ifndef _UINT8_CL_
#define _UINT8_CL_

#include "dtypes/bool.cl"

typedef uchar dt_uint8;

static inline dt_uint8 dt_zero_uint8() { return 0; }
static inline dt_uint8 dt_one_uint8() { return 1; }


dt_uint8 dt_add_uint8(dt_uint8 x, dt_uint8 y) {
    return x + y;
}

dt_uint8 dt_subtract_uint8(dt_uint8 x, dt_uint8 y) {
    return x - y;
}

dt_uint8 dt_multiply_uint8(dt_uint8 x, dt_uint8 y) {
    return x * y;
}

dt_uint8 dt_divide_uint8(dt_uint8 x, dt_uint8 y) {
    return x / y;
}


dt_uint8 dt_floor_divide_uint8(dt_uint8 x, dt_uint8 y) {
    return x / y;
}

dt_uint8 dt_power_uint8(dt_uint8 x, dt_uint8 y)
{
    if (x == 0)
        return y == 0;

    dt_uint8 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}


dt_uint8 dt_bitwise_and_uint8(dt_uint8 x, dt_uint8 y) {
    return x & y;
}

dt_uint8 dt_bitwise_or_uint8(dt_uint8 x, dt_uint8 y) {
    return x | y;
}

dt_uint8 dt_bitwise_xor_uint8(dt_uint8 x, dt_uint8 y) {
    return x ^ y;
}

dt_uint8 dt_left_shift_uint8(dt_uint8 x, dt_uint8 y) {
    return y >= (sizeof(y)*8) ? 0 : x << y;
}

dt_uint8 dt_right_shift_uint8(dt_uint8 x, dt_uint8 y) {
    return y >= (sizeof(y)*8) ? 0 : x >> y;
}

dt_uint8 dt_negative_uint8(dt_uint8 x) {
    return -x;
}

dt_uint8 dt_positive_uint8(dt_uint8 x) {
    return x;
}

dt_uint8 dt_bitwise_not_uint8(dt_uint8 x) {
    return ~x;
}

dt_uint8 dt_square_uint8(dt_uint8 x) {
    return x * x;
}

dt_uint8 dt_abs_uint8(dt_uint8 x) {
    return x;
}

dt_uint8 dt_sign_uint8(dt_uint8 x) {
    return x > 0;
}

dt_bool dt_equal_uint8(dt_uint8 x, dt_uint8 y) {
    return x == y;
}

dt_bool dt_not_equal_uint8(dt_uint8 x, dt_uint8 y) {
    return x != y;
}

dt_bool dt_greater_uint8(dt_uint8 x, dt_uint8 y) {
    return x > y;
}

dt_bool dt_greater_equal_uint8(dt_uint8 x, dt_uint8 y) {
    return x >= y;
}

dt_bool dt_less_uint8(dt_uint8 x, dt_uint8 y) {
    return x < y;
}

dt_bool dt_less_equal_uint8(dt_uint8 x, dt_uint8 y) {
    return x <= y;
}


dt_bool dt_logical_and_uint8(dt_uint8 x, dt_uint8 y) {
    return x && y;
}

dt_bool dt_logical_or_uint8(dt_uint8 x, dt_uint8 y) {
    return x || y;
}

dt_bool dt_logical_xor_uint8(dt_uint8 x, dt_uint8 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_uint8(dt_uint8 x) {
    return x == 0;
}

dt_uint8 dt_min_uint8(dt_uint8 x, dt_uint8 y) {
    return min(x, y);
}

dt_uint8 dt_max_uint8(dt_uint8 x, dt_uint8 y) {
    return max(x, y);
}

dt_uint8 dt_mad_uint8(dt_uint8 x, dt_uint8 y, dt_uint8 z) {
    return x * y + z;
}

#endif