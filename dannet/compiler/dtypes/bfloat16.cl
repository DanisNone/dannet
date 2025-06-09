#ifndef _BFLOAT16_CL_
#define _BFLOAT16_CL_
#include "dtypes/bool.cl"


typedef ushort dt_bfloat16;
typedef ushort dt_bfloat16_bits;
typedef float dt_bfloat16_work;

float normalize_bfloat16_input(dt_bfloat16 h) {
    return as_float(((uint)h) << 16);
}

dt_bfloat16 normalize_bfloat16_output(float x) {
    uint bits = *(uint*)(&x);
    uint exp = (bits >> 23) & 0xFF;

    if (exp == 0xFF) {
        return (ushort)(bits >> 16);
    }

    uint round_bit = (bits >> 15) & 1;
    uint lsb = (bits >> 16) & 1;
    uint lower_bits = bits & 0xFFFF;

    if (lower_bits > 0x8000 || (lower_bits == 0x8000 && lsb == 1)) {
        bits += 0x10000;
    }

    return (ushort)(bits >> 16);
}



static inline dt_bfloat16 dt_zero_bfloat16() { return 0; }
static inline dt_bfloat16 dt_one_bfloat16() { return normalize_bfloat16_output((dt_bfloat16_work)1); }

__constant dt_bfloat16_work dt_const_log2_bfloat16 = 0.6931471805599453;
__constant dt_bfloat16_work dt_const_log10_bfloat16 = 2.302585092994046;

static inline dt_bfloat16 dt_add_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(
        normalize_bfloat16_input(x) + 
        normalize_bfloat16_input(y)
    );
}

static inline dt_bfloat16 dt_subtract_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(
        normalize_bfloat16_input(x) - 
        normalize_bfloat16_input(y)
    );
}

static inline dt_bfloat16 dt_multiply_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(
        normalize_bfloat16_input(x) * 
        normalize_bfloat16_input(y)
    );
}

static inline dt_bfloat16 dt_divide_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(
        normalize_bfloat16_input(x) / 
        normalize_bfloat16_input(y)
    );
}

static inline dt_bfloat16 dt_min_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(fmin(
        normalize_bfloat16_input(x),
        normalize_bfloat16_input(y)
    ));
}

static inline dt_bfloat16 dt_max_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(fmax(
        normalize_bfloat16_input(x),
        normalize_bfloat16_input(y)
    ));
}

static inline dt_bfloat16 dt_mad_bfloat16(dt_bfloat16 x, dt_bfloat16 y, dt_bfloat16 z) {
    return normalize_bfloat16_output(
        normalize_bfloat16_input(x) *
        normalize_bfloat16_input(y) + 
        normalize_bfloat16_input(z)
    );
}

static inline dt_bfloat16 dt_floor_divide_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(floor(
        normalize_bfloat16_input(x) / 
        normalize_bfloat16_input(y)
    ));
}

static inline dt_bfloat16 dt_power_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(pow(
        normalize_bfloat16_input(x),
        normalize_bfloat16_input(y)
    ));
}

static inline dt_bfloat16 dt_arctan2_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    return normalize_bfloat16_output(atan2(
        normalize_bfloat16_input(x),
        normalize_bfloat16_input(y)
    ));
}

static inline dt_bfloat16 dt_logaddexp_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    dt_bfloat16_work xn = normalize_bfloat16_input(x);
    dt_bfloat16_work yn = normalize_bfloat16_input(y);
    if (xn < yn) {dt_bfloat16_work tmp = xn; xn = yn; yn = tmp;}
    return normalize_bfloat16_output(
        xn + 
        log1p(exp(yn - xn))
    );
}

static inline dt_bfloat16 dt_logaddexp2_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {
    dt_bfloat16_work xn = normalize_bfloat16_input(x);
    dt_bfloat16_work yn = normalize_bfloat16_input(y);
    if (xn < yn) {dt_bfloat16_work tmp = xn; xn = yn; yn = tmp;}

    return normalize_bfloat16_output(
        xn + 
        log1p(exp2(yn - xn)) / dt_const_log2_bfloat16
    );
}

static inline dt_bfloat16 dt_square_bfloat16(dt_bfloat16 x) {
    dt_bfloat16_work xn = normalize_bfloat16_input(x);
    return normalize_bfloat16_output(xn * xn);
}

static inline dt_bfloat16 dt_round_bfloat16(dt_bfloat16 x) {
    dt_bfloat16_work xn = normalize_bfloat16_input(x);
    
    dt_bfloat16_work rounded = round(xn);

    if (fabs(xn - rounded) == (dt_bfloat16_work)0.5) {
        rounded = ((dt_bfloat16_work)2.0) * round(xn * (dt_bfloat16_work)0.5);
    }
    return normalize_bfloat16_output(rounded);
}



static inline dt_bool dt_logical_not_bfloat16(dt_bfloat16 x) {
    dt_bfloat16_work xn = normalize_bfloat16_input(x);
    return !(dt_bool)xn;
}

#define _make_bfloat16_eq_func(func, op) \
static inline dt_bool dt_##func##_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {\
    return normalize_bfloat16_input(x) op normalize_bfloat16_input(y);\
}

#define _make_bfloat16_logical_func(func, op) \
static inline dt_bool dt_##func##_bfloat16(dt_bfloat16 x, dt_bfloat16 y) {\
    return ((dt_bool)normalize_bfloat16_input(x)) op ((dt_bool)normalize_bfloat16_input(y));\
}

#define _make_bfloat16_func(func, clfunc) \
static inline dt_bfloat16 dt_##func##_bfloat16(dt_bfloat16 x) {\
    return normalize_bfloat16_output(clfunc(\
        normalize_bfloat16_input(x)\
    ));\
}


_make_bfloat16_func(negative, -)
_make_bfloat16_func(abs, fabs)
_make_bfloat16_func(sqrt, sqrt)
_make_bfloat16_func(rsqrt, 1.0 / sqrt)


_make_bfloat16_func(exp, exp)
_make_bfloat16_func(exp2, exp2)
_make_bfloat16_func(exp10, exp10)
_make_bfloat16_func(expm1, expm1)

_make_bfloat16_func(log, log)
_make_bfloat16_func(log2, log2)
_make_bfloat16_func(log10, log10)
_make_bfloat16_func(log1p, log1p)

_make_bfloat16_func(sign, sign)
_make_bfloat16_func(sin, sin)
_make_bfloat16_func(cos, cos)
_make_bfloat16_func(tan, tan)
_make_bfloat16_func(sinh, sinh)
_make_bfloat16_func(cosh, cosh)
_make_bfloat16_func(tanh, tanh)
_make_bfloat16_func(arcsin, asin)
_make_bfloat16_func(arccos, acos)
_make_bfloat16_func(arctan, atan)
_make_bfloat16_func(arcsinh, asinh)
_make_bfloat16_func(arccosh, acosh)
_make_bfloat16_func(arctanh, atanh)


_make_bfloat16_func(floor, floor)
_make_bfloat16_func(ceil, ceil)
_make_bfloat16_func(trunc, trunc)


_make_bfloat16_eq_func(equal, ==)
_make_bfloat16_eq_func(not_equal, !=)
_make_bfloat16_eq_func(greater, >)
_make_bfloat16_eq_func(greater_equal, >=)
_make_bfloat16_eq_func(less, <)
_make_bfloat16_eq_func(less_equal, <=)

_make_bfloat16_logical_func(logical_and, &&)
_make_bfloat16_logical_func(logical_or, ||)
_make_bfloat16_logical_func(logical_xor, !=)


#endif