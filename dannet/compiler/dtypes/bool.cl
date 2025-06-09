#ifndef _BOOL_CL_
#define _BOOL_CL_


typedef bool dt_bool;


dt_bool dt_zero_bool() {
    return false;
}
dt_bool dt_one_bool() {
    return true;
}

dt_bool dt_add_bool(dt_bool x, dt_bool y) {
    return x | y;
}

dt_bool dt_subtract_bool(dt_bool x, dt_bool y) {
    return x ^ y;
}

dt_bool dt_multiply_bool(dt_bool x, dt_bool y) {
    return x & y;
}

dt_bool dt_divide_bool(dt_bool x, dt_bool y) {
    // x / 0 == UB == 0;
    // x / 1       == x
    return x & y;
}

dt_bool dt_floor_divide_bool(dt_bool x, dt_bool y) {
    // x / 0 == UB == 0;
    // x / 1       == x
    return x & y;
}

dt_bool dt_power_bool(dt_bool x, dt_bool y) {
    // 0^0 == 1
    // 0^1 == 0
    // 1^0 == 1
    // 1^1 == 1
    return x >= y;
}

dt_bool dt_left_shift_bool(dt_bool x, dt_bool y) {
    // 0 << 0 == 0
    // 1 << 0 == 1
    // 0 << 1 == 0
    // 1 << 1 == 2 == 1
    return x;
}


dt_bool dt_right_shift_bool(dt_bool x, dt_bool y) {
    // 0 >> 0 == 0
    // 1 >> 0 == 1
    // 0 >> 1 == 0
    // 1 >> 1 == 0
    return x > y;
}

dt_bool dt_negative_bool(dt_bool x) {
    // 0 => -0 => 0
    // 1 => -1 => 1
    return x;
}

dt_bool dt_bitwise_not_bool(dt_bool x) {
    return !x;
}

dt_bool dt_square_bool(dt_bool x) {
    return x;
}

dt_bool dt_abs_bool(dt_bool x) {
    return x;
}

dt_bool dt_sign_bool(dt_bool x) {
    return x;
}

dt_bool dt_equal_bool(dt_bool x, dt_bool y) {
    return x == y;
}

dt_bool dt_not_equal_bool(dt_bool x, dt_bool y) {
    return x != y;
}

dt_bool dt_greater_bool(dt_bool x, dt_bool y) {
    return x > y;
}

dt_bool dt_greater_equal_bool(dt_bool x, dt_bool y) {
    return x >= y;
}

dt_bool dt_less_bool(dt_bool x, dt_bool y) {
    return x < y;
}

dt_bool dt_less_equal_bool(dt_bool x, dt_bool y) {
    return x <= y;
}

dt_bool dt_bitwise_and_bool(dt_bool x, dt_bool y) {
    return x & y;
}

dt_bool dt_bitwise_or_bool(dt_bool x, dt_bool y) {
    return x | y;
}

dt_bool dt_bitwise_xor_bool(dt_bool x, dt_bool y) {
    return x ^ y;
}


dt_bool dt_logical_and_bool(dt_bool x, dt_bool y) {
    return x & y;
}

dt_bool dt_logical_or_bool(dt_bool x, dt_bool y) {
    return x | y;
}

dt_bool dt_logical_xor_bool(dt_bool x, dt_bool y) {
    return x ^ y;
}


dt_bool dt_logical_not_bool(dt_bool x) {
    return !x;
}

dt_bool dt_min_bool(dt_bool x, dt_bool y) {
    return x & y;
}

dt_bool dt_max_bool(dt_bool x, dt_bool y) {
    return x | y;
}

dt_bool dt_mad_bool(dt_bool x, dt_bool y, dt_bool z) {
    return (dt_bool)(x * y + z);
}

#endif