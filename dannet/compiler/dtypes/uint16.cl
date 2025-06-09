#ifndef _UINT16_CL_
#define _UINT16_CL_

#include "dtypes/bool.cl"

typedef ushort dt_uint16;

static inline dt_uint16 dt_zero_uint16() { return 0; }
static inline dt_uint16 dt_one_uint16() { return 1; }


dt_uint16 dt_add_uint16(dt_uint16 x, dt_uint16 y) {
    return x + y;
}

dt_uint16 dt_subtract_uint16(dt_uint16 x, dt_uint16 y) {
    return x - y;
}

dt_uint16 dt_multiply_uint16(dt_uint16 x, dt_uint16 y) {
    return x * y;
}

dt_uint16 dt_divide_uint16(dt_uint16 x, dt_uint16 y) {
    return x / y;
}


dt_uint16 dt_floor_divide_uint16(dt_uint16 x, dt_uint16 y) {
    return x / y;
}

dt_uint16 dt_power_uint16(dt_uint16 x, dt_uint16 y)
{
    if (x == 0)
        return y == 0;

    dt_uint16 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}


dt_uint16 dt_bitwise_and_uint16(dt_uint16 x, dt_uint16 y) {
    return x & y;
}

dt_uint16 dt_bitwise_or_uint16(dt_uint16 x, dt_uint16 y) {
    return x | y;
}

dt_uint16 dt_bitwise_xor_uint16(dt_uint16 x, dt_uint16 y) {
    return x ^ y;
}

dt_uint16 dt_left_shift_uint16(dt_uint16 x, dt_uint16 y) {
    return y >= (sizeof(y)*8) ? 0 : x << y;
}

dt_uint16 dt_right_shift_uint16(dt_uint16 x, dt_uint16 y) {
    return y >= (sizeof(y)*8) ? 0 : x >> y;
}

dt_uint16 dt_negative_uint16(dt_uint16 x) {
    return -x;
}

dt_uint16 dt_bitwise_not_uint16(dt_uint16 x) {
    return ~x;
}

dt_uint16 dt_square_uint16(dt_uint16 x) {
    return x * x;
}

dt_uint16 dt_abs_uint16(dt_uint16 x) {
    return x;
}

dt_uint16 dt_sign_uint16(dt_uint16 x) {
    return x > 0;
}

dt_bool dt_equal_uint16(dt_uint16 x, dt_uint16 y) {
    return x == y;
}

dt_bool dt_not_equal_uint16(dt_uint16 x, dt_uint16 y) {
    return x != y;
}

dt_bool dt_greater_uint16(dt_uint16 x, dt_uint16 y) {
    return x > y;
}

dt_bool dt_greater_equal_uint16(dt_uint16 x, dt_uint16 y) {
    return x >= y;
}

dt_bool dt_less_uint16(dt_uint16 x, dt_uint16 y) {
    return x < y;
}

dt_bool dt_less_equal_uint16(dt_uint16 x, dt_uint16 y) {
    return x <= y;
}


dt_bool dt_logical_and_uint16(dt_uint16 x, dt_uint16 y) {
    return x && y;
}

dt_bool dt_logical_or_uint16(dt_uint16 x, dt_uint16 y) {
    return x || y;
}

dt_bool dt_logical_xor_uint16(dt_uint16 x, dt_uint16 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_uint16(dt_uint16 x) {
    return x == 0;
}

dt_uint16 dt_min_uint16(dt_uint16 x, dt_uint16 y) {
    return min(x, y);
}

dt_uint16 dt_max_uint16(dt_uint16 x, dt_uint16 y) {
    return max(x, y);
}

dt_uint16 dt_mad_uint16(dt_uint16 x, dt_uint16 y, dt_uint16 z) {
    return x * y + z;
}

#endif