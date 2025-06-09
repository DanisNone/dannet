#ifndef _INT64_CL_
#define _INT64_CL_


#include "dtypes/bool.cl"

typedef long dt_int64;

static inline dt_int64 dt_zero_int64() { return 0; }
static inline dt_int64 dt_one_int64() { return 1; }


dt_int64 dt_add_int64(dt_int64 x, dt_int64 y) {
    return x + y;
}

dt_int64 dt_subtract_int64(dt_int64 x, dt_int64 y) {
    return x - y;
}

dt_int64 dt_multiply_int64(dt_int64 x, dt_int64 y) {
    return x * y;
}

dt_int64 dt_divide_int64(dt_int64 x, dt_int64 y) {
    return x / y;
}

dt_int64 dt_floor_divide_int64(dt_int64 x, dt_int64 y) {
    dt_int64 q = x / y;
    dt_int64 r = x % y;

    return q - (dt_bool)((r != 0) && ((x ^ y) < 0));
}

dt_int64 dt_power_int64(dt_int64 x, dt_int64 y)
{
    if (y < 0)
        return x == 1;
    if (x == 0)
        return y == 0;

    dt_int64 res = 1;
    while (y > 0)
    {
        if (y & 1)
            res *= x;
        x *= x;
        y >>= 1;
    }
    return res;
}

dt_int64 dt_bitwise_and_int64(dt_int64 x, dt_int64 y) {
    return x & y;
}

dt_int64 dt_bitwise_or_int64(dt_int64 x, dt_int64 y) {
    return x | y;
}

dt_int64 dt_bitwise_xor_int64(dt_int64 x, dt_int64 y) {
    return x ^ y;
}

dt_int64 dt_left_shift_int64(dt_int64 x, dt_int64 y) {
    return y >= (8*sizeof(y)) ? 0 : x << y;
}

dt_int64 dt_right_shift_int64(dt_int64 x, dt_int64 y) {
    return y >= (8*sizeof(y)) ? (-(x < 0)) : x >> y;
}

dt_int64 dt_negative_int64(dt_int64 x) {
    return -x;
}



dt_int64 dt_bitwise_not_int64(dt_int64 x) {
    return ~x;
}

dt_int64 dt_square_int64(dt_int64 x) {
    return x * x;
}


dt_int64 dt_abs_int64(dt_int64 x) {
    return abs(x);
}

dt_int64 dt_sign_int64(dt_int64 x) {
    return (x > 0) - (x < 0);
}


dt_bool dt_equal_int64(dt_int64 x, dt_int64 y) {
    return x == y;
}

dt_bool dt_not_equal_int64(dt_int64 x, dt_int64 y) {
    return x != y;
}

dt_bool dt_greater_int64(dt_int64 x, dt_int64 y) {
    return x > y;
}

dt_bool dt_greater_equal_int64(dt_int64 x, dt_int64 y) {
    return x >= y;
}

dt_bool dt_less_int64(dt_int64 x, dt_int64 y) {
    return x < y;
}

dt_bool dt_less_equal_int64(dt_int64 x, dt_int64 y) {
    return x <= y;
}

dt_bool dt_logical_and_int64(dt_int64 x, dt_int64 y) {
    return x && y;
}

dt_bool dt_logical_or_int64(dt_int64 x, dt_int64 y) {
    return x || y;
}

dt_bool dt_logical_xor_int64(dt_int64 x, dt_int64 y) {
    return (x != 0) ^ (y != 0);
}

dt_bool dt_logical_not_int64(dt_int64 x) {
    return x == 0;
}

dt_int64 dt_min_int64(dt_int64 x, dt_int64 y) {
    return min(x, y);
}

dt_int64 dt_max_int64(dt_int64 x, dt_int64 y) {
    return max(x, y);
}

dt_int64 dt_mad_int64(dt_int64 x, dt_int64 y, dt_int64 z) {
    return x * y + z;
}

#endif