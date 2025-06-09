#ifndef _INT16_CL_
#define _INT16_CL_


#include "dtypes/bool.cl"

typedef short dt_int16;

static inline dt_int16 dt_zero_int16() { return 0; }
static inline dt_int16 dt_one_int16() { return 1; }


dt_int16 dt_add_int16(dt_int16 x, dt_int16 y) {
    return x + y;
}

dt_int16 dt_subtract_int16(dt_int16 x, dt_int16 y) {
    return x - y;
}

dt_int16 dt_multiply_int16(dt_int16 x, dt_int16 y) {
    return x * y;
}

dt_int16 dt_divide_int16(dt_int16 x, dt_int16 y) {
    return x / y;
}

dt_int16 dt_floor_divide_int16(dt_int16 x, dt_int16 y) {
    dt_int16 q = x / y;
    dt_int16 r = x % y;

    return q - (dt_bool)((r != 0) && ((x ^ y) < 0));
}

dt_int16 dt_power_int16(dt_int16 x, dt_int16 y)
{
    if (y < 0)
        return x == 1;
    if (x == 0)
        return y == 0;

    dt_int16 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}

dt_int16 dt_bitwise_and_int16(dt_int16 x, dt_int16 y) {
    return x & y;
}

dt_int16 dt_bitwise_or_int16(dt_int16 x, dt_int16 y) {
    return x | y;
}

dt_int16 dt_bitwise_xor_int16(dt_int16 x, dt_int16 y) {
    return x ^ y;
}

dt_int16 dt_left_shift_int16(dt_int16 x, dt_int16 y) {
    return y >= (8*sizeof(y)) ? 0 : x << y;
}

dt_int16 dt_right_shift_int16(dt_int16 x, dt_int16 y) {
    return y >= (8*sizeof(y)) ? (-(x < 0)) : x >> y;
}

dt_int16 dt_negative_int16(dt_int16 x) {
    return -x;
}

dt_int16 dt_positive_int16(dt_int16 x) {
    return x;
}


dt_int16 dt_bitwise_not_int16(dt_int16 x) {
    return ~x;
}

dt_int16 dt_square_int16(dt_int16 x) {
    return x * x;
}


dt_int16 dt_abs_int16(dt_int16 x) {
    return abs(x);
}

dt_int16 dt_sign_int16(dt_int16 x) {
    return (x > 0) - (x < 0);
}


dt_bool dt_equal_int16(dt_int16 x, dt_int16 y) {
    return x == y;
}

dt_bool dt_not_equal_int16(dt_int16 x, dt_int16 y) {
    return x != y;
}

dt_bool dt_greater_int16(dt_int16 x, dt_int16 y) {
    return x > y;
}

dt_bool dt_greater_equal_int16(dt_int16 x, dt_int16 y) {
    return x >= y;
}

dt_bool dt_less_int16(dt_int16 x, dt_int16 y) {
    return x < y;
}

dt_bool dt_less_equal_int16(dt_int16 x, dt_int16 y) {
    return x <= y;
}

dt_bool dt_logical_and_int16(dt_int16 x, dt_int16 y) {
    return x && y;
}

dt_bool dt_logical_or_int16(dt_int16 x, dt_int16 y) {
    return x || y;
}

dt_bool dt_logical_xor_int16(dt_int16 x, dt_int16 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_int16(dt_int16 x) {
    return x == 0;
}

dt_int16 dt_min_int16(dt_int16 x, dt_int16 y) {
    return min(x, y);
}

dt_int16 dt_max_int16(dt_int16 x, dt_int16 y) {
    return max(x, y);
}

dt_int16 dt_mad_int16(dt_int16 x, dt_int16 y, dt_int16 z) {
    return x * y + z;
}

#endif