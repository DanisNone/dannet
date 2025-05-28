#ifndef _FLOAT16_CL_
#define _FLOAT16_CL_

#include "dtypes/bool.cl"
#ifdef cl_khr_fp16
typedef ushort dt_float16;
typedef uint dt_float16_bits;
typedef float dt_float16_work;

float normalize_float16_input(dt_float16 h) {
    uint s = (h >> 15) & 0x00000001;
    uint e = (h >> 10) & 0x0000001F;
    uint f = h & 0x000003FF;

    uint out_e, out_f;

    if (e == 0) {
        if (f == 0) {
            out_e = 0;
            out_f = 0;
        } else {
            e = 1;
            while ((f & 0x00000400) == 0) {
                f <<= 1;
                e--;
            }
            f &= 0x000003FF;
            out_e = 127 - 15 - e;
            out_f = f << 13;
        }
    } else if (e == 31) {
        out_e = 255;
        out_f = f << 13;
    } else {
        out_e = e + (127 - 15);
        out_f = f << 13;
    }

    uint result = (s << 31) | (out_e << 23) | out_f;
    return as_float(result);
}

dt_float16 normalize_float16_output(float x) {
    uint i = as_uint(x);
    uint s = (i >> 31) & 0x1;
    int e = ((i >> 23) & 0xFF) - 127 + 15;
    uint f = i & 0x007FFFFF;

    ushort h;

    if ((i & 0x7FFFFFFF) == 0) {
        h = (ushort)(s << 15);
    }
    else if (((i >> 23) & 0xFF) == 0xFF) {
        if (f == 0) {
            h = (ushort)((s << 15) | (0x1F << 10));
        } else {
            h = (ushort)((s << 15) | (0x1F << 10) | (f >> 13));
        }
    }
    else if (e <= 0) {
        if (e < -10) {
            h = (ushort)(s << 15);
        } else {
            f = (f | 0x00800000) >> (1 - e);
            h = (ushort)((s << 15) | (f >> 13));
        }
    }
    else if (e >= 31) {
        h = (ushort)((s << 15) | (0x1F << 10));
    }
    else {
        h = (ushort)((s << 15) | (e << 10) | (f >> 13));
    }

    return h;
}
#else
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half dt_float16;
typedef half dt_float16_work;
typedef ushort dt_float16_bits;

static inline dt_float16 normalize_float16_input(dt_float16 x)  { return x; }
static inline dt_float16 normalize_float16_output(dt_float16 x) { return x; }
#endif


static inline dt_float16 dt_zero_float16() { return 0; }
static inline dt_float16 dt_one_float16() { return normalize_float16_output((dt_float16_work)1); }

__constant dt_float16_work dt_const_log2_float16 = 0.6931471805599453;
__constant dt_float16_work dt_const_log10_float16 = 2.302585092994046;

static inline dt_float16 dt_add_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(
        normalize_float16_input(x) + 
        normalize_float16_input(y)
    );
}

static inline dt_float16 dt_subtract_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(
        normalize_float16_input(x) - 
        normalize_float16_input(y)
    );
}

static inline dt_float16 dt_multiply_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(
        normalize_float16_input(x) * 
        normalize_float16_input(y)
    );
}

static inline dt_float16 dt_divide_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(
        normalize_float16_input(x) / 
        normalize_float16_input(y)
    );
}

static inline dt_float16 dt_min_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(fmin(
        normalize_float16_input(x),
        normalize_float16_input(y)
    ));
}

static inline dt_float16 dt_max_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(fmax(
        normalize_float16_input(x),
        normalize_float16_input(y)
    ));
}

static inline dt_float16 dt_mad_float16(dt_float16 x, dt_float16 y, dt_float16 z) {
    return normalize_float16_output(
        normalize_float16_input(x) *
        normalize_float16_input(y) + 
        normalize_float16_input(z)
    );
}

static inline dt_float16 dt_floor_divide_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(floor(
        normalize_float16_input(x) / 
        normalize_float16_input(y)
    ));
}

static inline dt_float16 dt_power_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(pow(
        normalize_float16_input(x),
        normalize_float16_input(y)
    ));
}

static inline dt_float16 dt_arctan2_float16(dt_float16 x, dt_float16 y) {
    return normalize_float16_output(atan2(
        normalize_float16_input(x),
        normalize_float16_input(y)
    ));
}

static inline dt_float16 dt_logaddexp_float16(dt_float16 x, dt_float16 y) {
    dt_float16_work xn = normalize_float16_input(x);
    dt_float16_work yn = normalize_float16_input(y);
    if (xn < yn) {dt_float16_work tmp = xn; xn = yn; yn = tmp;}
    return normalize_float16_output(
        xn + 
        log1p(exp(yn - xn))
    );
}

static inline dt_float16 dt_logaddexp2_float16(dt_float16 x, dt_float16 y) {
    dt_float16_work xn = normalize_float16_input(x);
    dt_float16_work yn = normalize_float16_input(y);
    if (xn < yn) {dt_float16_work tmp = xn; xn = yn; yn = tmp;}

    return normalize_float16_output(
        xn + 
        log1p(exp2(yn - xn)) / dt_const_log2_float16
    );
}

static inline dt_float16 dt_square_float16(dt_float16 x) {
    dt_float16_work xn = normalize_float16_input(x);
    return normalize_float16_output(xn * xn);
}

static inline dt_float16 dt_round_float16(dt_float16 x) {
    dt_float16_work xn = normalize_float16_input(x);
    
    dt_float16_work rounded = round(xn);

    if (fabs(xn - rounded) == (dt_float16_work)0.5) {
        rounded = ((dt_float16_work)2.0) * round(xn * (dt_float16_work)0.5);
    }
    return normalize_float16_output(rounded);
}



static inline dt_bool dt_logical_not_float16(dt_float16 x) {
    dt_float16_work xn = normalize_float16_input(x);
    return !(dt_bool)xn;
}

#define _make_float16_eq_func(func, op) \
static inline dt_bool dt_##func##_float16(dt_float16 x, dt_float16 y) {\
    return normalize_float16_input(x) op normalize_float16_input(y);\
}

#define _make_float16_logical_func(func, op) \
static inline dt_bool dt_##func##_float16(dt_float16 x, dt_float16 y) {\
    return ((dt_bool)normalize_float16_input(x)) op ((dt_bool)normalize_float16_input(y));\
}

#define _make_float16_func(func, clfunc) \
static inline dt_float16 dt_##func##_float16(dt_float16 x) {\
    return normalize_float16_output(clfunc(\
        normalize_float16_input(x)\
    ));\
}


_make_float16_func(negative, -)
_make_float16_func(abs, fabs)
_make_float16_func(sqrt, sqrt)
_make_float16_func(rsqrt, 1.0 / sqrt)


_make_float16_func(exp, exp)
_make_float16_func(exp2, exp2)
_make_float16_func(exp10, exp10)
_make_float16_func(expm1, expm1)

_make_float16_func(log, log)
_make_float16_func(log2, log2)
_make_float16_func(log10, log10)
_make_float16_func(log1p, log1p)

_make_float16_func(sign, sign)
_make_float16_func(sin, sin)
_make_float16_func(cos, cos)
_make_float16_func(tan, tan)
_make_float16_func(sinh, sinh)
_make_float16_func(cosh, cosh)
_make_float16_func(tanh, tanh)
_make_float16_func(arcsin, asin)
_make_float16_func(arccos, acos)
_make_float16_func(arctan, atan)
_make_float16_func(arcsinh, asinh)
_make_float16_func(arccosh, acosh)
_make_float16_func(arctanh, atanh)


_make_float16_func(floor, floor)
_make_float16_func(ceil, ceil)
_make_float16_func(trunc, trunc)


_make_float16_eq_func(equal, ==)
_make_float16_eq_func(not_equal, !=)
_make_float16_eq_func(greater, >)
_make_float16_eq_func(greater_equal, >=)
_make_float16_eq_func(less, <)
_make_float16_eq_func(less_equal, <=)

_make_float16_logical_func(logical_and, &&)
_make_float16_logical_func(logical_or, ||)
_make_float16_logical_func(logical_xor, !=)


#endif