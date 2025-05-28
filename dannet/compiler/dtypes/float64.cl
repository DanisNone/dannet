#ifndef _FLOAT64_CL_
#define _FLOAT64_CL_
#include "dtypes/bool.cl"

#ifndef cl_khr_fp64
typedef ulong dt_float64;
typedef uint dt_float64_bits;
typedef float dt_float64_work;

__constant dt_float64 dt_const_log2_float64 = 0x3fe62e42fefa39efUL;
__constant dt_float64 dt_const_log10_float64 = 0x40026bb1bbb55516UL;

float normalize_float64_input(dt_float64 h) {
    uint sign     = (h >> 63) & 0x1;
    int  exponent = (int)((h >> 52) & 0x7FF);
    ulong mantissa = h & 0xFFFFFFFFFFFFFUL;

    if (exponent == 0x7FF) {
        uint f_sign = sign << 31;
        if (mantissa != 0) {
            return as_float(f_sign | 0x7FC00000);
        } else {
            return as_float(f_sign | 0x7F800000);
        }
    }

    int f_exp;
    uint f_mant;

    if (exponent == 0) {
        f_exp = 0;
        f_mant = 0;
    } else {
        exponent -= 1023;
        exponent += 127;

        if (exponent <= 0) {
            f_exp = 0;
            f_mant = 0;
        } else if (exponent >= 0xFF) {
            return as_float((sign << 31) | 0x7F800000);
        } else {
            f_exp = exponent;
            ulong full_mant = mantissa | (1UL << 52);
            f_mant = (uint)(full_mant >> (52 - 23));
        }
    }

    uint floatBits = (sign << 31) | ((f_exp & 0xFF) << 23) | (f_mant & 0x7FFFFF);
    return as_float(floatBits);
}

dt_float64 normalize_float64_output(float f) {
    uint i = as_uint(f);
    uint sign = (i >> 31) & 0x1;
    int exp = (int)((i >> 23) & 0xFF);
    uint frac = i & 0x7FFFFF;

    ulong out_sign = ((ulong)sign) << 63;
    int out_exp;
    ulong out_frac;

    if (exp == 0) {
        if (frac == 0) {
            return out_sign;
        } else {
            int shift = 0;
            uint mant = frac;
            while ((mant & 0x400000) == 0) {
                mant <<= 1;
                shift++;
            }
            mant &= 0x3FFFFF;
            out_exp = 1023 - 127 - shift + 1;
            out_frac = ((ulong)mant) << (52 - 23);
            return out_sign | ((ulong)out_exp << 52) | out_frac;
        }
    } else if (exp == 0xFF) {
        out_exp = 0x7FF;
        out_frac = (frac != 0) ? (((ulong)frac) << (52 - 23)) | 0x0008000000000000UL : 0;
        return out_sign | ((ulong)out_exp << 52) | out_frac;
    } else {
        out_exp = exp - 127 + 1023;
        out_frac = ((ulong)frac) << (52 - 23);
        return out_sign | ((ulong)out_exp << 52) | out_frac;
    }
}
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double dt_float64;
typedef ulong dt_float64_bits;
typedef double dt_float64_work;

static inline dt_float64 normalize_float64_input(dt_float64 x)  { return x; }
static inline dt_float64 normalize_float64_output(dt_float64 x) { return x; }
__constant dt_float64 dt_const_log2_float64 = 0.6931471805599453;
__constant dt_float64 dt_const_log10_float64 = 2.302585092994046;
#endif


static inline dt_float64 dt_zero_float64() { return 0; }
static inline dt_float64 dt_one_float64() { return normalize_float64_output((dt_float64_work)1); }


static inline dt_float64 dt_add_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(
        normalize_float64_input(x) + 
        normalize_float64_input(y)
    );
}

static inline dt_float64 dt_subtract_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(
        normalize_float64_input(x) - 
        normalize_float64_input(y)
    );
}

static inline dt_float64 dt_multiply_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(
        normalize_float64_input(x) * 
        normalize_float64_input(y)
    );
}

static inline dt_float64 dt_divide_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(
        normalize_float64_input(x) / 
        normalize_float64_input(y)
    );
}

static inline dt_float64 dt_min_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(fmin(
        normalize_float64_input(x),
        normalize_float64_input(y)
    ));
}

static inline dt_float64 dt_max_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(fmax(
        normalize_float64_input(x),
        normalize_float64_input(y)
    ));
}

static inline dt_float64 dt_mad_float64(dt_float64 x, dt_float64 y, dt_float64 z) {
    return normalize_float64_output(
        normalize_float64_input(x) *
        normalize_float64_input(y) + 
        normalize_float64_input(z)
    );
}

static inline dt_float64 dt_floor_divide_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(floor(
        normalize_float64_input(x) / 
        normalize_float64_input(y)
    ));
}

static inline dt_float64 dt_power_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(pow(
        normalize_float64_input(x),
        normalize_float64_input(y)
    ));
}

static inline dt_float64 dt_arctan2_float64(dt_float64 x, dt_float64 y) {
    return normalize_float64_output(atan2(
        normalize_float64_input(x),
        normalize_float64_input(y)
    ));
}

static inline dt_float64 dt_logaddexp_float64(dt_float64 x, dt_float64 y) {
    dt_float64_work xn = normalize_float64_input(x);
    dt_float64_work yn = normalize_float64_input(y);
    if (xn < yn) {dt_float64_work tmp = xn; xn = yn; yn = tmp;}
    return normalize_float64_output(
        xn + 
        log1p(exp(yn - xn))
    );
}

static inline dt_float64 dt_logaddexp2_float64(dt_float64 x, dt_float64 y) {
    dt_float64_work xn = normalize_float64_input(x);
    dt_float64_work yn = normalize_float64_input(y);
    if (xn < yn) {dt_float64_work tmp = xn; xn = yn; yn = tmp;}

    return normalize_float64_output(
        xn + 
        log1p(exp2(yn - xn)) / dt_const_log2_float64
    );
}

static inline dt_float64 dt_square_float64(dt_float64 x) {
    dt_float64_work xn = normalize_float64_input(x);
    return normalize_float64_output(xn * xn);
}

static inline dt_float64 dt_round_float64(dt_float64 x) {
    dt_float64_work xn = normalize_float64_input(x);
    
    dt_float64_work rounded = round(xn);

    if (fabs(xn - rounded) == (dt_float64_work)0.5) {
        rounded = ((dt_float64_work)2.0) * round(xn * (dt_float64_work)0.5);
    }
    return normalize_float64_output(rounded);
}



static inline dt_bool dt_logical_not_float64(dt_float64 x) {
    dt_float64_work xn = normalize_float64_input(x);
    return !(dt_bool)xn;
}

#define _make_float64_eq_func(func, op) \
static inline dt_bool dt_##func##_float64(dt_float64 x, dt_float64 y) {\
    return normalize_float64_input(x) op normalize_float64_input(y);\
}

#define _make_float64_logical_func(func, op) \
static inline dt_bool dt_##func##_float64(dt_float64 x, dt_float64 y) {\
    return ((dt_bool)normalize_float64_input(x)) op ((dt_bool)normalize_float64_input(y));\
}

#define _make_float64_func(func, clfunc) \
static inline dt_float64 dt_##func##_float64(dt_float64 x) {\
    return normalize_float64_output(clfunc(\
        normalize_float64_input(x)\
    ));\
}


_make_float64_func(negative, -)
_make_float64_func(abs, fabs)
_make_float64_func(sqrt, sqrt)
_make_float64_func(rsqrt, 1.0 / sqrt)


_make_float64_func(exp, exp)
_make_float64_func(exp2, exp2)
_make_float64_func(exp10, exp10)
_make_float64_func(expm1, expm1)

_make_float64_func(log, log)
_make_float64_func(log2, log2)
_make_float64_func(log10, log10)
_make_float64_func(log1p, log1p)

_make_float64_func(sign, sign)
_make_float64_func(sin, sin)
_make_float64_func(cos, cos)
_make_float64_func(tan, tan)
_make_float64_func(sinh, sinh)
_make_float64_func(cosh, cosh)
_make_float64_func(tanh, tanh)
_make_float64_func(arcsin, asin)
_make_float64_func(arccos, acos)
_make_float64_func(arctan, atan)
_make_float64_func(arcsinh, asinh)
_make_float64_func(arccosh, acosh)
_make_float64_func(arctanh, atanh)


_make_float64_func(floor, floor)
_make_float64_func(ceil, ceil)
_make_float64_func(trunc, trunc)


_make_float64_eq_func(equal, ==)
_make_float64_eq_func(not_equal, !=)
_make_float64_eq_func(greater, >)
_make_float64_eq_func(greater_equal, >=)
_make_float64_eq_func(less, <)
_make_float64_eq_func(less_equal, <=)

_make_float64_logical_func(logical_and, &&)
_make_float64_logical_func(logical_or, ||)
_make_float64_logical_func(logical_xor, !=)


#endif