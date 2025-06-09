#ifndef _COMPLEX128_CL_
#define _COMPLEX128_CL_


#include "dtypes/float64.cl"
#include "dtypes/bool.cl"

typedef struct {
    dt_float64 real, imag;
} dt_complex128;


static inline dt_complex128 make_complex128(dt_float64 real, dt_float64 imag) {
    dt_complex128 res;
    res.real = real;
    res.imag = imag;

    return res;
}

static inline dt_complex128 make_complex128_from_work(dt_float64_work real, dt_float64_work imag) {
    dt_complex128 res;
    res.real = normalize_float64_output(real);
    res.imag = normalize_float64_output(imag);

    return res;
}

static inline dt_complex128 dt_zero_complex128() { return make_complex128(0, 0); }
static inline dt_complex128 dt_one_complex128() { return make_complex128(dt_one_float64(), 0); }

static inline dt_complex128 dt_add_complex128(dt_complex128 x, dt_complex128 y) { 
    return make_complex128_from_work(
        normalize_float64_input(x.real) + normalize_float64_input(y.real),
        normalize_float64_input(x.imag) + normalize_float64_input(y.imag)
    ); 
}


static inline dt_complex128 dt_subtract_complex128(dt_complex128 x, dt_complex128 y) { 
    return make_complex128_from_work(
        normalize_float64_input(x.real) - normalize_float64_input(y.real),
        normalize_float64_input(x.imag) - normalize_float64_input(y.imag)
    ); 
}

static inline dt_complex128 dt_multiply_complex128(dt_complex128 x, dt_complex128 y) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    return make_complex128_from_work(
        x_real * y_real - x_imag * y_imag,
        x_real * y_imag + x_imag * y_real
    ); 
}


static inline dt_complex128 dt_divide_complex128(dt_complex128 x, dt_complex128 y) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    dt_float64_work norm2 = y_real * y_real + y_imag * y_imag;

    return make_complex128_from_work(
        (x_real * y_real + x_imag * y_imag) / norm2,
        (x_imag * y_real - x_real * y_imag) / norm2
    ); 
}

static inline dt_complex128 dt_power_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    dt_float64_work log_real = log(x_real * x_real + x_imag * x_imag) / 2;
    dt_float64_work log_imag = atan2(x_imag, x_real);
    
    dt_float64_work exp_input_imag = (y_real * log_imag + y_imag * log_real);
    
    dt_float64_work r = exp(y_real * log_real - y_imag * log_imag);
    dt_float64_work cos_ = cos(exp_input_imag);
    dt_float64_work sin_ = sin(exp_input_imag);
    
    return make_complex128_from_work(
        cos_ * r,
        sin_ * r
    );
}

static inline dt_complex128 dt_min_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);    
    dt_float64_work y_real = normalize_float64_input(y.real);
    
    return x_real < y_real ? x : y;
}

static inline dt_complex128 dt_max_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);    
    dt_float64_work y_real = normalize_float64_input(y.real);
    
    return x_real > y_real ? x : y;
}

static inline dt_complex128 dt_negative_complex128(dt_complex128 x) { 
    return make_complex128_from_work(
        -normalize_float64_input(x.real),
        -normalize_float64_input(x.imag)
    );
}

static inline dt_complex128 dt_positive_complex128(dt_complex128 x) { 
    return x;
}

static inline dt_complex128 dt_square_complex128(dt_complex128 x) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    return make_complex128_from_work(
        x_real * x_real - x_imag * x_imag,
        2 * x_real * x_imag
    ); 
}

static inline dt_complex128 dt_mad_complex128(dt_complex128 x, dt_complex128 y, dt_complex128 z) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);

    dt_float64_work z_real = normalize_float64_input(z.real);
    dt_float64_work z_imag = normalize_float64_input(z.imag);

    return make_complex128_from_work(
        x_real * y_real - x_imag * y_imag + z_real,
        x_real * y_imag + x_imag * y_real + z_imag
    );
}

static inline dt_complex128 dt_round_complex128(dt_complex128 x) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    
    dt_float64_work rounded_real = round(x_real);
    dt_float64_work rounded_imag = round(x_imag);

    if (fabs(x_real - rounded_real) == (dt_float64_work)0.5) {
        rounded_real = ((dt_float64_work)2.0) * round(x_real * (dt_float64_work)0.5);
    }
    if (fabs(x_imag - rounded_imag) == (dt_float64_work)0.5) {
        rounded_imag = ((dt_float64_work)2.0) * round(x_imag * (dt_float64_work)0.5);
    }

    return make_complex128_from_work(
        rounded_real,
        rounded_imag
    );
}

static inline dt_bool dt_logical_not_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    return !((dt_bool)x_real || (dt_bool)x_imag);
}

static inline dt_complex128 dt_sqrt_complex128(dt_complex128 x) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work magnitude = sqrt(x_real * x_real + x_imag * x_imag);
    dt_float64_work real_part = sqrt((magnitude + x_real) / 2);

    dt_float64_work imag_part = sign(x_imag) * sqrt((magnitude - x_real) / 2);
    return make_complex128_from_work(real_part, imag_part);
}

static inline dt_complex128 dt_rsqrt_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work magnitude = sqrt(x_real * x_real + x_imag * x_imag);
    
    dt_float64_work half_angle = -atan2(x_imag, x_real) / 2;

    dt_float64_work rsqrt_mag = 1.0 / sqrt(magnitude);

    dt_float64_work real_part = rsqrt_mag * cos(half_angle);
    dt_float64_work imag_part = rsqrt_mag * sin(half_angle);

    return make_complex128_from_work(real_part, imag_part);
}

inline dt_complex128 dt_log_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        log(x_real*x_real + x_imag*x_imag) / 2,
        atan2(x_imag, x_real)
    );
}

inline dt_complex128 dt_log2_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        log2(x_real*x_real + x_imag*x_imag) / 2,
        atan2(x_imag, x_real) / dt_const_log2_float64
    );
}

inline dt_complex128 dt_log10_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        log10(x_real*x_real + x_imag*x_imag) / 2,
        atan2(x_imag, x_real) / dt_const_log10_float64
    );
}

inline dt_complex128 dt_log1p_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real) + 1;
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        log(x_real*x_real + x_imag*x_imag) / 2,
        atan2(x_imag, x_real)
    );
}

inline dt_complex128 dt_logaddexp_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    dt_float64_work m = fmax(x_real, y_real);

    dt_float64_work r1 = exp(x_real - m);
    dt_float64_work cos1 = cos(x_imag);
    dt_float64_work sin1 = sin(x_imag);

    dt_float64_work r2 = exp(y_real - m);
    dt_float64_work cos2 = cos(y_imag);
    dt_float64_work sin2 = sin(y_imag);

    dt_float64_work z_real = r1 * cos1 + r2 * cos2;
    dt_float64_work z_imag = r1 * sin1 + r2 * sin2;

    return make_complex128_from_work(
        m + log(z_real * z_real + z_imag * z_imag) / 2,
        atan2(z_imag, z_real)
    );
}

inline dt_complex128 dt_logaddexp2_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag) * dt_const_log2_float64;

    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag) * dt_const_log2_float64;
    
    dt_float64_work m = fmax(x_real, y_real);

    dt_float64_work r1 = exp2(x_real - m);
    dt_float64_work cos1 = cos(x_imag);
    dt_float64_work sin1 = sin(x_imag);

    dt_float64_work r2 = exp2(y_real - m);
    dt_float64_work cos2 = cos(y_imag);
    dt_float64_work sin2 = sin(y_imag);

    dt_float64_work z_real = r1 * cos1 + r2 * cos2;
    dt_float64_work z_imag = r1 * sin1 + r2 * sin2;

    return make_complex128_from_work(
        m + log2(z_real * z_real + z_imag * z_imag) / 2,
        atan2(z_imag, z_real) / dt_const_log2_float64
    );
}

inline dt_complex128 dt_exp_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64 cos_ = cos(x_imag);
    dt_float64 sin_ = sin(x_imag);
    dt_float64 r   = exp(x_real);
    return make_complex128_from_work(
        r * cos_,
        r * sin_
    );
}

inline dt_complex128 dt_exp2_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag) * dt_const_log2_float64;
    
    dt_float64 cos_ = cos(x_imag);
    dt_float64 sin_ = sin(x_imag);
    dt_float64 r   = exp2(x_real);
    return make_complex128_from_work(
        r * cos_,
        r * sin_
    );
}

inline dt_complex128 dt_exp10_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag) * dt_const_log10_float64;
    
    dt_float64 cos_ = cos(x_imag);
    dt_float64 sin_ = sin(x_imag);
    dt_float64 r   = exp10(x_real);
    return make_complex128_from_work(
        r * cos_,
        r * sin_
    );
}

inline dt_complex128 dt_expm1_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64 cos_ = cos(x_imag);
    dt_float64 sin_ = sin(x_imag);
    dt_float64 r   = exp(x_real);
    return make_complex128_from_work(
        r * cos_ - 1,
        r * sin_
    );
}

static inline dt_float64 dt_abs_complex128(dt_complex128 x) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return normalize_float64_output(sqrt(
        x_real * x_real + x_imag * x_imag
    ));
}

static inline dt_complex128 dt_sign_complex128(dt_complex128 x) { 
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    dt_float64_work norm = sqrt(x_real * x_real + x_imag * x_imag);
    return make_complex128_from_work(
        x_real / norm,
        x_imag / norm
    ); 
}

inline dt_complex128 dt_sin_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        sin(x_real) * cosh(x_imag),
        cos(x_real) * sinh(x_imag)
    );
}

inline dt_complex128 dt_cos_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        cos(x_real) * cosh(x_imag),
        -sin(x_real) * sinh(x_imag)
    );
}

inline dt_complex128 dt_tan_complex128(dt_complex128 x) {
    dt_float64_work x_real2 = 2*normalize_float64_input(x.real);
    dt_float64_work x_imag2 = 2*normalize_float64_input(x.imag);
        
    dt_float64_work sum = cos(x_real2) + cosh(x_imag2);
    return make_complex128_from_work(
        sin(x_real2) / sum,
        sinh(x_imag2) / sum
    );
}


inline dt_complex128 dt_sinh_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        cos(x_imag) * sinh(x_real),
        sin(x_imag) * cosh(x_real)
    );
}

inline dt_complex128 dt_cosh_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);
    
    return make_complex128_from_work(
        cos(x_imag) * cosh(x_real),
        sin(x_imag) * sinh(x_real)
    );
}

inline dt_complex128 dt_tanh_complex128(dt_complex128 x) {
    dt_float64_work x_real2 = 2*normalize_float64_input(x.real);
    dt_float64_work x_imag2 = 2*normalize_float64_input(x.imag);
    
    dt_float64_work sum = cos(x_imag2) + cosh(x_real2);
    return make_complex128_from_work(
        sinh(x_real2) / sum,
        sin(x_imag2) / sum
    );
}

inline dt_complex128 dt_arcsin_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    // compute sqrt

    dt_float64_work sqrt_in_real = 1 - x_real * x_real + x_imag * x_imag;
    dt_float64_work sqrt_in_imag = -2 * x_real * x_imag;
    
    dt_float64_work sqrt_magnitude = sqrt(sqrt_in_real * sqrt_in_real + sqrt_in_imag * sqrt_in_imag);

    dt_float64_work sqrt_real_part = sqrt((sqrt_magnitude + sqrt_in_real) / 2);
    dt_float64_work sqrt_imag_part = sign(sqrt_in_imag) * sqrt((sqrt_magnitude - sqrt_in_real) / 2);
    
    // compute log

    dt_float64_work log_in_real = sqrt_real_part - x_imag;
    dt_float64_work log_in_imag = x_real + sqrt_imag_part;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 2;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real);
    return make_complex128_from_work(
        log_imag_part,
        -log_real_part
    );
}

inline dt_complex128 dt_arccos_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    // compute sqrt

    dt_float64_work sqrt_in_real = 1 - x_real * x_real + x_imag * x_imag;
    dt_float64_work sqrt_in_imag = -2 * x_real * x_imag;
    
    dt_float64_work sqrt_magnitude = sqrt(sqrt_in_real * sqrt_in_real + sqrt_in_imag * sqrt_in_imag);

    dt_float64_work sqrt_real_part = sqrt((sqrt_magnitude + sqrt_in_real) / 2);
    dt_float64_work sqrt_imag_part = sign(sqrt_in_imag) * sqrt((sqrt_magnitude - sqrt_in_real) / 2);
    
    // compute log

    dt_float64_work log_in_real = x_real - sqrt_imag_part;
    dt_float64_work log_in_imag = x_imag + sqrt_real_part;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 2;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real);
    return make_complex128_from_work(
        log_imag_part,
        -log_real_part
    );
}

inline dt_complex128 dt_arctan_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work x_real2 = x_real * x_real;
    dt_float64_work x_imag2 = x_imag * x_imag;

    // compute divide

    dt_float64_work denom = (x_imag - 1) * (x_imag - 1) + x_real2;
    dt_float64_work div_real_part = 1 - x_real2 - x_imag2; 
    dt_float64_work div_imag_part = -2 * x_real; 
    
    // compute log

    dt_float64_work log_in_real = div_real_part / denom;
    dt_float64_work log_in_imag = div_imag_part / denom;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 4;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real) / 2;
    return make_complex128_from_work(
        -log_imag_part,
        log_real_part
    );
}

inline dt_complex128 dt_arctan2_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work y_real = normalize_float64_input(y.real);
    dt_float64_work y_imag = normalize_float64_input(y.imag);


    dt_float64_work a = y_real + x_imag;
    dt_float64_work b = y_imag - x_real;
    dt_float64_work c = y_real - x_imag;
    dt_float64_work d = y_imag + x_real;

    dt_float64_work norm2 = c*c + d*d;

    dt_float64_work u = a*c + b*d;
    dt_float64_work v = b*c - a*d;

    return make_complex128_from_work(
        -0.5 * atan2(v, u),
        0.25 * log((u*u + v*v) / (norm2 * norm2))
    );
}

inline dt_complex128 dt_arcsinh_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    // compute sqrt

    dt_float64_work sqrt_in_real = 1 + x_real * x_real - x_imag * x_imag;
    dt_float64_work sqrt_in_imag = 2 * x_real * x_imag;
    
    dt_float64_work sqrt_magnitude = sqrt(sqrt_in_real * sqrt_in_real + sqrt_in_imag * sqrt_in_imag);

    dt_float64_work sqrt_real_part = sqrt((sqrt_magnitude + sqrt_in_real) / 2);
    dt_float64_work sqrt_imag_part = sign(sqrt_in_imag) * sqrt((sqrt_magnitude - sqrt_in_real) / 2);
    
    // compute log

    dt_float64_work log_in_real = sqrt_real_part + x_real;
    dt_float64_work log_in_imag = sqrt_imag_part + x_imag;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 2;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real);
    return make_complex128_from_work(
        log_real_part,
        log_imag_part
    );
}


inline dt_complex128 dt_arccosh_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    // compute sqrt

    dt_float64_work sqrt_in_real = x_real * x_real - x_imag * x_imag - 1;
    dt_float64_work sqrt_in_imag = 2 * x_real * x_imag;
    
    dt_float64_work sqrt_magnitude = sqrt(sqrt_in_real * sqrt_in_real + sqrt_in_imag * sqrt_in_imag);

    dt_float64_work sqrt_real_part = sqrt((sqrt_magnitude + sqrt_in_real) / 2);
    dt_float64_work sqrt_imag_part = sign(sqrt_in_imag) * sqrt((sqrt_magnitude - sqrt_in_real) / 2);
    
    // compute log

    dt_float64_work log_in_real = sqrt_real_part + x_real;
    dt_float64_work log_in_imag = sqrt_imag_part + x_imag;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 2;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real);
    
    return make_complex128_from_work(
        fabs(log_real_part),
        log_imag_part * sign(log_real_part)
    );
}

inline dt_complex128 dt_arctanh_complex128(dt_complex128 x) {
    dt_float64_work x_real = normalize_float64_input(x.real);
    dt_float64_work x_imag = normalize_float64_input(x.imag);

    dt_float64_work x_real2 = x_real * x_real;
    dt_float64_work x_imag2 = x_imag * x_imag;

    // compute divide

    dt_float64_work denom = (x_real - 1) * (x_real - 1) + x_imag2;
    dt_float64_work div_real_part = 1 - x_real2 - x_imag2; 
    dt_float64_work div_imag_part = 2 * x_imag; 
    
    // compute log

    dt_float64_work log_in_real = div_real_part / denom;
    dt_float64_work log_in_imag = div_imag_part / denom;

    dt_float64_work log_real_part = log(log_in_real*log_in_real + log_in_imag*log_in_imag) / 4;
    dt_float64_work log_imag_part = atan2(log_in_imag, log_in_real) / 2;
    return make_complex128_from_work(
        log_real_part,
        log_imag_part
    );
}


static inline dt_bool dt_equal_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);    
    dt_float64_work y_real = normalize_float64_input(y.real);    

    if (x_real != y_real)
        return false;

    dt_float64_work x_imag = normalize_float64_input(x.imag);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    return x_imag == y_imag;
}

static inline dt_bool dt_not_equal_complex128(dt_complex128 x, dt_complex128 y) {
    dt_float64_work x_real = normalize_float64_input(x.real);    
    dt_float64_work y_real = normalize_float64_input(y.real);    

    if (x_real == y_real)
        return false;

    dt_float64_work x_imag = normalize_float64_input(x.imag);
    dt_float64_work y_imag = normalize_float64_input(y.imag);
    
    return x_imag != y_imag;
}

static inline dt_complex128 dt_conjugate_complex128(dt_complex128 x) {
    return make_complex128(
        x.real,
        normalize_float64_output(-normalize_float64_input(x.imag))
    );
}

static inline dt_float64 dt_real_complex128(dt_complex128 x) {
    return x.real;
}

static inline dt_float64 dt_imag_complex128(dt_complex128 x) {
    return x.imag;
}

#define _make_complex128_eq_func(func, op) \
static inline dt_bool dt_##func##_complex128(dt_complex128 x, dt_complex128 y) {\
    return normalize_float64_input(x.real) op normalize_float64_input(y.real);\
}

#define _make_complex128_logical_func(func, op) \
static inline dt_bool dt_##func##_complex128(dt_complex128 x, dt_complex128 y) {\
    return ((dt_bool)(normalize_float64_input(x.real) || normalize_float64_input(x.imag))) op \
           ((dt_bool)(normalize_float64_input(y.real) || normalize_float64_input(y.imag)));\
}

_make_complex128_eq_func(greater, >)
_make_complex128_eq_func(greater_equal, >=)

_make_complex128_eq_func(less, <)
_make_complex128_eq_func(less_equal, <=)

_make_complex128_logical_func(logical_and, &&)
_make_complex128_logical_func(logical_or, ||)
_make_complex128_logical_func(logical_xor, !=)



#endif