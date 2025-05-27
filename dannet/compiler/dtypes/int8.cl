#ifndef _INT8_CL_
#define _INT8_CL_


#include "dtypes/bool.cl"

typedef char dt_int8;

static inline dt_int8 dt_zero_int8() { return 0; }
static inline dt_int8 dt_one_int8() { return 1; }


dt_int8 dt_add_int8(dt_int8 x, dt_int8 y) {
    return x + y;
}

dt_int8 dt_subtract_int8(dt_int8 x, dt_int8 y) {
    return x - y;
}

dt_int8 dt_multiply_int8(dt_int8 x, dt_int8 y) {
    return x * y;
}

dt_int8 dt_divide_int8(dt_int8 x, dt_int8 y) {
    return x / y;
}

dt_int8 dt_floor_divide_int8(dt_int8 x, dt_int8 y) {
    dt_int8 q = x / y;
    dt_int8 r = x % y;

    return q - (dt_bool)((r != 0) && ((x ^ y) < 0));
}

dt_int8 dt_power_int8(dt_int8 x, dt_int8 y)
{
    if (y < 0)
        return x == 1;
    if (x == 0)
        return y == 0;

    dt_int8 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}

dt_int8 dt_bitwise_and_int8(dt_int8 x, dt_int8 y) {
    return x & y;
}

dt_int8 dt_bitwise_or_int8(dt_int8 x, dt_int8 y) {
    return x | y;
}

dt_int8 dt_bitwise_xor_int8(dt_int8 x, dt_int8 y) {
    return x ^ y;
}

dt_int8 dt_left_shift_int8(dt_int8 x, dt_int8 y) {
    return y >= (8*sizeof(y)) ? 0 : x << y;
}

dt_int8 dt_right_shift_int8(dt_int8 x, dt_int8 y) {
    return y >= (8*sizeof(y)) ? (-(x < 0)) : x >> y;
}

dt_int8 dt_negative_int8(dt_int8 x) {
    return -x;
}



dt_int8 dt_bitwise_not_int8(dt_int8 x) {
    return ~x;
}

dt_int8 dt_square_int8(dt_int8 x) {
    return x * x;
}


dt_int8 dt_abs_int8(dt_int8 x) {
    return abs(x);
}

dt_int8 dt_sign_int8(dt_int8 x) {
    return (x > 0) - (x < 0);
}


dt_bool dt_equal_int8(dt_int8 x, dt_int8 y) {
    return x == y;
}

dt_bool dt_not_equal_int8(dt_int8 x, dt_int8 y) {
    return x != y;
}

dt_bool dt_greater_int8(dt_int8 x, dt_int8 y) {
    return x > y;
}

dt_bool dt_greater_equal_int8(dt_int8 x, dt_int8 y) {
    return x >= y;
}

dt_bool dt_less_int8(dt_int8 x, dt_int8 y) {
    return x < y;
}

dt_bool dt_less_equal_int8(dt_int8 x, dt_int8 y) {
    return x <= y;
}

dt_bool dt_logical_and_int8(dt_int8 x, dt_int8 y) {
    return x && y;
}

dt_bool dt_logical_or_int8(dt_int8 x, dt_int8 y) {
    return x || y;
}

dt_bool dt_logical_xor_int8(dt_int8 x, dt_int8 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_int8(dt_int8 x) {
    return x == 0;
}

dt_int8 dt_min_int8(dt_int8 x, dt_int8 y) {
    return min(x, y);
}

dt_int8 dt_max_int8(dt_int8 x, dt_int8 y) {
    return max(x, y);
}

dt_int8 dt_mad_int8(dt_int8 x, dt_int8 y, dt_int8 z) {
    return x * y + z;
}

#endif