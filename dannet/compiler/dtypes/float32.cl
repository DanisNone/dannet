#ifndef _FLOAT32_CL_
#define _FLOAT32_CL_

#include "dtypes/bool.cl"

typedef float dt_float32;
typedef uint dt_float32_bits;
typedef float dt_float32_work;


__constant dt_float32_work dt_const_log2_float32 = 0.6931471805599453;
__constant dt_float32_work dt_const_log10_float32 = 2.302585092994046;

static inline dt_float32 normalize_float32_input(dt_float32 x)  { return x; }
static inline dt_float32 normalize_float32_output(dt_float32 x) { return x; }
static inline dt_float32 dt_zero_float32() { return 0; }
static inline dt_float32 dt_one_float32() { return normalize_float32_output((dt_float32_work)1); }


static inline dt_float32 dt_add_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(
        normalize_float32_input(x) + 
        normalize_float32_input(y)
    );
}

static inline dt_float32 dt_subtract_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(
        normalize_float32_input(x) - 
        normalize_float32_input(y)
    );
}

static inline dt_float32 dt_multiply_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(
        normalize_float32_input(x) * 
        normalize_float32_input(y)
    );
}

static inline dt_float32 dt_divide_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(
        normalize_float32_input(x) / 
        normalize_float32_input(y)
    );
}

static inline dt_float32 dt_min_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(fmin(
        normalize_float32_input(x),
        normalize_float32_input(y)
    ));
}

static inline dt_float32 dt_max_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(fmax(
        normalize_float32_input(x),
        normalize_float32_input(y)
    ));
}

static inline dt_float32 dt_mad_float32(dt_float32 x, dt_float32 y, dt_float32 z) {
    return normalize_float32_output(
        normalize_float32_input(x) *
        normalize_float32_input(y) + 
        normalize_float32_input(z)
    );
}

static inline dt_float32 dt_floor_divide_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(floor(
        normalize_float32_input(x) / 
        normalize_float32_input(y)
    ));
}

static inline dt_float32 dt_power_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(pow(
        normalize_float32_input(x),
        normalize_float32_input(y)
    ));
}

static inline dt_float32 dt_arctan2_float32(dt_float32 x, dt_float32 y) {
    return normalize_float32_output(atan2(
        normalize_float32_input(x),
        normalize_float32_input(y)
    ));
}

static inline dt_float32 dt_logaddexp_float32(dt_float32 x, dt_float32 y) {
    dt_float32_work xn = normalize_float32_input(x);
    dt_float32_work yn = normalize_float32_input(y);
    if (xn < yn) {dt_float32_work tmp = xn; xn = yn; yn = tmp;}
    return normalize_float32_output(
        xn + 
        log1p(exp(yn - xn))
    );
}

static inline dt_float32 dt_logaddexp2_float32(dt_float32 x, dt_float32 y) {
    dt_float32_work xn = normalize_float32_input(x);
    dt_float32_work yn = normalize_float32_input(y);
    if (xn < yn) {dt_float32_work tmp = xn; xn = yn; yn = tmp;}

    return normalize_float32_output(
        xn + 
        log1p(exp2(yn - xn)) / dt_const_log2_float32
    );
}

static inline dt_float32 dt_square_float32(dt_float32 x) {
    dt_float32_work xn = normalize_float32_input(x);
    return normalize_float32_output(xn * xn);
}

static inline dt_float32 dt_round_float32(dt_float32 x) {
    dt_float32_work xn = normalize_float32_input(x);
    
    dt_float32_work rounded = round(xn);

    if (fabs(xn - rounded) == (dt_float32_work)0.5) {
        rounded = ((dt_float32_work)2.0) * round(xn * (dt_float32_work)0.5);
    }
    return normalize_float32_output(rounded);
}



static inline dt_bool dt_logical_not_float32(dt_float32 x) {
    dt_float32_work xn = normalize_float32_input(x);
    return !(dt_bool)xn;
}

#define _make_float32_eq_func(func, op) \
static inline dt_bool dt_##func##_float32(dt_float32 x, dt_float32 y) {\
    return normalize_float32_input(x) op normalize_float32_input(y);\
}

#define _make_float32_logical_func(func, op) \
static inline dt_bool dt_##func##_float32(dt_float32 x, dt_float32 y) {\
    return ((dt_bool)normalize_float32_input(x)) op ((dt_bool)normalize_float32_input(y));\
}

#define _make_float32_func(func, clfunc) \
static inline dt_float32 dt_##func##_float32(dt_float32 x) {\
    return normalize_float32_output(clfunc(\
        normalize_float32_input(x)\
    ));\
}


_make_float32_func(negative, -)
_make_float32_func(abs, fabs)
_make_float32_func(sqrt, sqrt)
_make_float32_func(rsqrt, 1.0 / sqrt)


_make_float32_func(exp, exp)
_make_float32_func(exp2, exp2)
_make_float32_func(exp10, exp10)
_make_float32_func(expm1, expm1)

_make_float32_func(log, log)
_make_float32_func(log2, log2)
_make_float32_func(log10, log10)
_make_float32_func(log1p, log1p)

_make_float32_func(sign, sign)
_make_float32_func(sin, sin)
_make_float32_func(cos, cos)
_make_float32_func(tan, tan)
_make_float32_func(sinh, sinh)
_make_float32_func(cosh, cosh)
_make_float32_func(tanh, tanh)
_make_float32_func(arcsin, asin)
_make_float32_func(arccos, acos)
_make_float32_func(arctan, atan)
_make_float32_func(arcsinh, asinh)
_make_float32_func(arccosh, acosh)
_make_float32_func(arctanh, atanh)


_make_float32_func(floor, floor)
_make_float32_func(ceil, ceil)
_make_float32_func(trunc, trunc)


_make_float32_eq_func(equal, ==)
_make_float32_eq_func(not_equal, !=)
_make_float32_eq_func(greater, >)
_make_float32_eq_func(greater_equal, >=)
_make_float32_eq_func(less, <)
_make_float32_eq_func(less_equal, <=)

_make_float32_logical_func(logical_and, &&)
_make_float32_logical_func(logical_or, ||)
_make_float32_logical_func(logical_xor, !=)


#endif