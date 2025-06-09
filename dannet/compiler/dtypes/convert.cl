#ifndef _CONVERT_CL_
#define _CONVERT_CL_

#include "dtypes/dtypes.cl"

static inline dt_bool dt_convert_bool_to_bool(dt_bool x) {
    return x;
}

static inline dt_bool dt_convert_uint8_to_bool(dt_uint8 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_uint16_to_bool(dt_uint16 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_uint32_to_bool(dt_uint32 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_uint64_to_bool(dt_uint64 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_int8_to_bool(dt_int8 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_int16_to_bool(dt_int16 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_int32_to_bool(dt_int32 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_int64_to_bool(dt_int64 x) {
    return (dt_bool)(x);
}

static inline dt_bool dt_convert_float16_to_bool(dt_float16 x) {
    return (dt_bool)(normalize_float16_input(x));
}

static inline dt_bool dt_convert_bfloat16_to_bool(dt_bfloat16 x) {
    return (dt_bool)(normalize_bfloat16_input(x));
}

static inline dt_bool dt_convert_float32_to_bool(dt_float32 x) {
    return (dt_bool)(normalize_float32_input(x));
}

static inline dt_bool dt_convert_float64_to_bool(dt_float64 x) {
    return (dt_bool)(normalize_float64_input(x));
}

static inline dt_bool dt_convert_complex64_to_bool(dt_complex64 x) {
    return ((dt_bool)(normalize_float32_input(x.real))) || ((dt_bool)(normalize_float32_input(x.imag)));
}

static inline dt_bool dt_convert_complex128_to_bool(dt_complex128 x) {
    return ((dt_bool)(normalize_float64_input(x.real))) || ((dt_bool)(normalize_float64_input(x.imag)));
}

static inline dt_uint8 dt_convert_bool_to_uint8(dt_bool x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_uint8_to_uint8(dt_uint8 x) {
    return x;
}

static inline dt_uint8 dt_convert_uint16_to_uint8(dt_uint16 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_uint32_to_uint8(dt_uint32 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_uint64_to_uint8(dt_uint64 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_int8_to_uint8(dt_int8 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_int16_to_uint8(dt_int16 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_int32_to_uint8(dt_int32 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_int64_to_uint8(dt_int64 x) {
    return (dt_uint8)x;
}

static inline dt_uint8 dt_convert_float16_to_uint8(dt_float16 x) {
    return (dt_uint8)(normalize_float16_input(x));
}

static inline dt_uint8 dt_convert_bfloat16_to_uint8(dt_bfloat16 x) {
    return (dt_uint8)(normalize_bfloat16_input(x));
}

static inline dt_uint8 dt_convert_float32_to_uint8(dt_float32 x) {
    return (dt_uint8)(normalize_float32_input(x));
}

static inline dt_uint8 dt_convert_float64_to_uint8(dt_float64 x) {
    return (dt_uint8)(normalize_float64_input(x));
}

static inline dt_uint8 dt_convert_complex64_to_uint8(dt_complex64 x) {
    return (dt_uint8)(normalize_float32_input(x.real));
}

static inline dt_uint8 dt_convert_complex128_to_uint8(dt_complex128 x) {
    return (dt_uint8)(normalize_float64_input(x.real));
}

static inline dt_uint16 dt_convert_bool_to_uint16(dt_bool x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_uint8_to_uint16(dt_uint8 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_uint16_to_uint16(dt_uint16 x) {
    return x;
}

static inline dt_uint16 dt_convert_uint32_to_uint16(dt_uint32 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_uint64_to_uint16(dt_uint64 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_int8_to_uint16(dt_int8 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_int16_to_uint16(dt_int16 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_int32_to_uint16(dt_int32 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_int64_to_uint16(dt_int64 x) {
    return (dt_uint16)x;
}

static inline dt_uint16 dt_convert_float16_to_uint16(dt_float16 x) {
    return (dt_uint16)(normalize_float16_input(x));
}

static inline dt_uint16 dt_convert_bfloat16_to_uint16(dt_bfloat16 x) {
    return (dt_uint16)(normalize_bfloat16_input(x));
}

static inline dt_uint16 dt_convert_float32_to_uint16(dt_float32 x) {
    return (dt_uint16)(normalize_float32_input(x));
}

static inline dt_uint16 dt_convert_float64_to_uint16(dt_float64 x) {
    return (dt_uint16)(normalize_float64_input(x));
}

static inline dt_uint16 dt_convert_complex64_to_uint16(dt_complex64 x) {
    return (dt_uint16)(normalize_float32_input(x.real));
}

static inline dt_uint16 dt_convert_complex128_to_uint16(dt_complex128 x) {
    return (dt_uint16)(normalize_float64_input(x.real));
}

static inline dt_uint32 dt_convert_bool_to_uint32(dt_bool x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_uint8_to_uint32(dt_uint8 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_uint16_to_uint32(dt_uint16 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_uint32_to_uint32(dt_uint32 x) {
    return x;
}

static inline dt_uint32 dt_convert_uint64_to_uint32(dt_uint64 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_int8_to_uint32(dt_int8 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_int16_to_uint32(dt_int16 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_int32_to_uint32(dt_int32 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_int64_to_uint32(dt_int64 x) {
    return (dt_uint32)x;
}

static inline dt_uint32 dt_convert_float16_to_uint32(dt_float16 x) {
    return (dt_uint32)(normalize_float16_input(x));
}

static inline dt_uint32 dt_convert_bfloat16_to_uint32(dt_bfloat16 x) {
    return (dt_uint32)(normalize_bfloat16_input(x));
}

static inline dt_uint32 dt_convert_float32_to_uint32(dt_float32 x) {
    return (dt_uint32)(normalize_float32_input(x));
}

static inline dt_uint32 dt_convert_float64_to_uint32(dt_float64 x) {
    return (dt_uint32)(normalize_float64_input(x));
}

static inline dt_uint32 dt_convert_complex64_to_uint32(dt_complex64 x) {
    return (dt_uint32)(normalize_float32_input(x.real));
}

static inline dt_uint32 dt_convert_complex128_to_uint32(dt_complex128 x) {
    return (dt_uint32)(normalize_float64_input(x.real));
}

static inline dt_uint64 dt_convert_bool_to_uint64(dt_bool x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_uint8_to_uint64(dt_uint8 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_uint16_to_uint64(dt_uint16 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_uint32_to_uint64(dt_uint32 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_uint64_to_uint64(dt_uint64 x) {
    return x;
}

static inline dt_uint64 dt_convert_int8_to_uint64(dt_int8 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_int16_to_uint64(dt_int16 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_int32_to_uint64(dt_int32 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_int64_to_uint64(dt_int64 x) {
    return (dt_uint64)x;
}

static inline dt_uint64 dt_convert_float16_to_uint64(dt_float16 x) {
    return (dt_uint64)(normalize_float16_input(x));
}

static inline dt_uint64 dt_convert_bfloat16_to_uint64(dt_bfloat16 x) {
    return (dt_uint64)(normalize_bfloat16_input(x));
}

static inline dt_uint64 dt_convert_float32_to_uint64(dt_float32 x) {
    return (dt_uint64)(normalize_float32_input(x));
}

static inline dt_uint64 dt_convert_float64_to_uint64(dt_float64 x) {
    return (dt_uint64)(normalize_float64_input(x));
}

static inline dt_uint64 dt_convert_complex64_to_uint64(dt_complex64 x) {
    return (dt_uint64)(normalize_float32_input(x.real));
}

static inline dt_uint64 dt_convert_complex128_to_uint64(dt_complex128 x) {
    return (dt_uint64)(normalize_float64_input(x.real));
}

static inline dt_int8 dt_convert_bool_to_int8(dt_bool x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_uint8_to_int8(dt_uint8 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_uint16_to_int8(dt_uint16 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_uint32_to_int8(dt_uint32 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_uint64_to_int8(dt_uint64 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_int8_to_int8(dt_int8 x) {
    return x;
}

static inline dt_int8 dt_convert_int16_to_int8(dt_int16 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_int32_to_int8(dt_int32 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_int64_to_int8(dt_int64 x) {
    return (dt_int8)x;
}

static inline dt_int8 dt_convert_float16_to_int8(dt_float16 x) {
    return (dt_int8)(normalize_float16_input(x));
}

static inline dt_int8 dt_convert_bfloat16_to_int8(dt_bfloat16 x) {
    return (dt_int8)(normalize_bfloat16_input(x));
}

static inline dt_int8 dt_convert_float32_to_int8(dt_float32 x) {
    return (dt_int8)(normalize_float32_input(x));
}

static inline dt_int8 dt_convert_float64_to_int8(dt_float64 x) {
    return (dt_int8)(normalize_float64_input(x));
}

static inline dt_int8 dt_convert_complex64_to_int8(dt_complex64 x) {
    return (dt_int8)(normalize_float32_input(x.real));
}

static inline dt_int8 dt_convert_complex128_to_int8(dt_complex128 x) {
    return (dt_int8)(normalize_float64_input(x.real));
}

static inline dt_int16 dt_convert_bool_to_int16(dt_bool x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_uint8_to_int16(dt_uint8 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_uint16_to_int16(dt_uint16 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_uint32_to_int16(dt_uint32 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_uint64_to_int16(dt_uint64 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_int8_to_int16(dt_int8 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_int16_to_int16(dt_int16 x) {
    return x;
}

static inline dt_int16 dt_convert_int32_to_int16(dt_int32 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_int64_to_int16(dt_int64 x) {
    return (dt_int16)x;
}

static inline dt_int16 dt_convert_float16_to_int16(dt_float16 x) {
    return (dt_int16)(normalize_float16_input(x));
}

static inline dt_int16 dt_convert_bfloat16_to_int16(dt_bfloat16 x) {
    return (dt_int16)(normalize_bfloat16_input(x));
}

static inline dt_int16 dt_convert_float32_to_int16(dt_float32 x) {
    return (dt_int16)(normalize_float32_input(x));
}

static inline dt_int16 dt_convert_float64_to_int16(dt_float64 x) {
    return (dt_int16)(normalize_float64_input(x));
}

static inline dt_int16 dt_convert_complex64_to_int16(dt_complex64 x) {
    return (dt_int16)(normalize_float32_input(x.real));
}

static inline dt_int16 dt_convert_complex128_to_int16(dt_complex128 x) {
    return (dt_int16)(normalize_float64_input(x.real));
}

static inline dt_int32 dt_convert_bool_to_int32(dt_bool x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_uint8_to_int32(dt_uint8 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_uint16_to_int32(dt_uint16 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_uint32_to_int32(dt_uint32 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_uint64_to_int32(dt_uint64 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_int8_to_int32(dt_int8 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_int16_to_int32(dt_int16 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_int32_to_int32(dt_int32 x) {
    return x;
}

static inline dt_int32 dt_convert_int64_to_int32(dt_int64 x) {
    return (dt_int32)x;
}

static inline dt_int32 dt_convert_float16_to_int32(dt_float16 x) {
    return (dt_int32)(normalize_float16_input(x));
}

static inline dt_int32 dt_convert_bfloat16_to_int32(dt_bfloat16 x) {
    return (dt_int32)(normalize_bfloat16_input(x));
}

static inline dt_int32 dt_convert_float32_to_int32(dt_float32 x) {
    return (dt_int32)(normalize_float32_input(x));
}

static inline dt_int32 dt_convert_float64_to_int32(dt_float64 x) {
    return (dt_int32)(normalize_float64_input(x));
}

static inline dt_int32 dt_convert_complex64_to_int32(dt_complex64 x) {
    return (dt_int32)(normalize_float32_input(x.real));
}

static inline dt_int32 dt_convert_complex128_to_int32(dt_complex128 x) {
    return (dt_int32)(normalize_float64_input(x.real));
}

static inline dt_int64 dt_convert_bool_to_int64(dt_bool x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_uint8_to_int64(dt_uint8 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_uint16_to_int64(dt_uint16 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_uint32_to_int64(dt_uint32 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_uint64_to_int64(dt_uint64 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_int8_to_int64(dt_int8 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_int16_to_int64(dt_int16 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_int32_to_int64(dt_int32 x) {
    return (dt_int64)x;
}

static inline dt_int64 dt_convert_int64_to_int64(dt_int64 x) {
    return x;
}

static inline dt_int64 dt_convert_float16_to_int64(dt_float16 x) {
    return (dt_int64)(normalize_float16_input(x));
}

static inline dt_int64 dt_convert_bfloat16_to_int64(dt_bfloat16 x) {
    return (dt_int64)(normalize_bfloat16_input(x));
}

static inline dt_int64 dt_convert_float32_to_int64(dt_float32 x) {
    return (dt_int64)(normalize_float32_input(x));
}

static inline dt_int64 dt_convert_float64_to_int64(dt_float64 x) {
    return (dt_int64)(normalize_float64_input(x));
}

static inline dt_int64 dt_convert_complex64_to_int64(dt_complex64 x) {
    return (dt_int64)(normalize_float32_input(x.real));
}

static inline dt_int64 dt_convert_complex128_to_int64(dt_complex128 x) {
    return (dt_int64)(normalize_float64_input(x.real));
}

static inline dt_float16 dt_convert_bool_to_float16(dt_bool x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_uint8_to_float16(dt_uint8 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_uint16_to_float16(dt_uint16 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_uint32_to_float16(dt_uint32 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_uint64_to_float16(dt_uint64 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_int8_to_float16(dt_int8 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_int16_to_float16(dt_int16 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_int32_to_float16(dt_int32 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_int64_to_float16(dt_int64 x) {
    return normalize_float16_output((dt_float16_work)x);
}

static inline dt_float16 dt_convert_float16_to_float16(dt_float16 x) {
    return x;
}

static inline dt_float16 dt_convert_bfloat16_to_float16(dt_bfloat16 x) {
    return normalize_float16_output(normalize_bfloat16_input(x));
}

static inline dt_float16 dt_convert_float32_to_float16(dt_float32 x) {
    return normalize_float16_output(normalize_float32_input(x));
}

static inline dt_float16 dt_convert_float64_to_float16(dt_float64 x) {
    return normalize_float16_output(normalize_float64_input(x));
}

static inline dt_float16 dt_convert_complex64_to_float16(dt_complex64 x) {
    return normalize_float16_output(normalize_float32_input(x.real));
}

static inline dt_float16 dt_convert_complex128_to_float16(dt_complex128 x) {
    return normalize_float16_output(normalize_float64_input(x.real));
}

static inline dt_bfloat16 dt_convert_bool_to_bfloat16(dt_bool x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_uint8_to_bfloat16(dt_uint8 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_uint16_to_bfloat16(dt_uint16 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_uint32_to_bfloat16(dt_uint32 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_uint64_to_bfloat16(dt_uint64 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_int8_to_bfloat16(dt_int8 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_int16_to_bfloat16(dt_int16 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_int32_to_bfloat16(dt_int32 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_int64_to_bfloat16(dt_int64 x) {
    return normalize_bfloat16_output((dt_bfloat16_work)x);
}

static inline dt_bfloat16 dt_convert_float16_to_bfloat16(dt_float16 x) {
    return normalize_bfloat16_output(normalize_float16_input(x));
}

static inline dt_bfloat16 dt_convert_bfloat16_to_bfloat16(dt_bfloat16 x) {
    return x;
}

static inline dt_bfloat16 dt_convert_float32_to_bfloat16(dt_float32 x) {
    return normalize_bfloat16_output(normalize_float32_input(x));
}

static inline dt_bfloat16 dt_convert_float64_to_bfloat16(dt_float64 x) {
    return normalize_bfloat16_output(normalize_float64_input(x));
}

static inline dt_bfloat16 dt_convert_complex64_to_bfloat16(dt_complex64 x) {
    return normalize_bfloat16_output(normalize_float32_input(x.real));
}

static inline dt_bfloat16 dt_convert_complex128_to_bfloat16(dt_complex128 x) {
    return normalize_bfloat16_output(normalize_float64_input(x.real));
}

static inline dt_float32 dt_convert_bool_to_float32(dt_bool x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_uint8_to_float32(dt_uint8 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_uint16_to_float32(dt_uint16 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_uint32_to_float32(dt_uint32 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_uint64_to_float32(dt_uint64 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_int8_to_float32(dt_int8 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_int16_to_float32(dt_int16 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_int32_to_float32(dt_int32 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_int64_to_float32(dt_int64 x) {
    return normalize_float32_output((dt_float32_work)x);
}

static inline dt_float32 dt_convert_float16_to_float32(dt_float16 x) {
    return normalize_float32_output(normalize_float16_input(x));
}

static inline dt_float32 dt_convert_bfloat16_to_float32(dt_bfloat16 x) {
    return normalize_float32_output(normalize_bfloat16_input(x));
}

static inline dt_float32 dt_convert_float32_to_float32(dt_float32 x) {
    return x;
}

static inline dt_float32 dt_convert_float64_to_float32(dt_float64 x) {
    return normalize_float32_output(normalize_float64_input(x));
}

static inline dt_float32 dt_convert_complex64_to_float32(dt_complex64 x) {
    return normalize_float32_output(normalize_float32_input(x.real));
}

static inline dt_float32 dt_convert_complex128_to_float32(dt_complex128 x) {
    return normalize_float32_output(normalize_float64_input(x.real));
}

static inline dt_float64 dt_convert_bool_to_float64(dt_bool x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_uint8_to_float64(dt_uint8 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_uint16_to_float64(dt_uint16 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_uint32_to_float64(dt_uint32 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_uint64_to_float64(dt_uint64 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_int8_to_float64(dt_int8 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_int16_to_float64(dt_int16 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_int32_to_float64(dt_int32 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_int64_to_float64(dt_int64 x) {
    return normalize_float64_output((dt_float64_work)x);
}

static inline dt_float64 dt_convert_float16_to_float64(dt_float16 x) {
    return normalize_float64_output(normalize_float16_input(x));
}

static inline dt_float64 dt_convert_bfloat16_to_float64(dt_bfloat16 x) {
    return normalize_float64_output(normalize_bfloat16_input(x));
}

static inline dt_float64 dt_convert_float32_to_float64(dt_float32 x) {
    return normalize_float64_output(normalize_float32_input(x));
}

static inline dt_float64 dt_convert_float64_to_float64(dt_float64 x) {
    return x;
}

static inline dt_float64 dt_convert_complex64_to_float64(dt_complex64 x) {
    return normalize_float64_output(normalize_float32_input(x.real));
}

static inline dt_float64 dt_convert_complex128_to_float64(dt_complex128 x) {
    return normalize_float64_output(normalize_float64_input(x.real));
}

static inline dt_complex64 dt_convert_bool_to_complex64(dt_bool x) {
    return make_complex64(dt_convert_bool_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_uint8_to_complex64(dt_uint8 x) {
    return make_complex64(dt_convert_uint8_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_uint16_to_complex64(dt_uint16 x) {
    return make_complex64(dt_convert_uint16_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_uint32_to_complex64(dt_uint32 x) {
    return make_complex64(dt_convert_uint32_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_uint64_to_complex64(dt_uint64 x) {
    return make_complex64(dt_convert_uint64_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_int8_to_complex64(dt_int8 x) {
    return make_complex64(dt_convert_int8_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_int16_to_complex64(dt_int16 x) {
    return make_complex64(dt_convert_int16_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_int32_to_complex64(dt_int32 x) {
    return make_complex64(dt_convert_int32_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_int64_to_complex64(dt_int64 x) {
    return make_complex64(dt_convert_int64_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_float16_to_complex64(dt_float16 x) {
    return make_complex64(dt_convert_float16_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_bfloat16_to_complex64(dt_bfloat16 x) {
    return make_complex64(dt_convert_bfloat16_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_float32_to_complex64(dt_float32 x) {
    return make_complex64(dt_convert_float32_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_float64_to_complex64(dt_float64 x) {
    return make_complex64(dt_convert_float64_to_float32(x), 0);
}

static inline dt_complex64 dt_convert_complex64_to_complex64(dt_complex64 x) {
    return x;
}

static inline dt_complex64 dt_convert_complex128_to_complex64(dt_complex128 x) {
    return make_complex64(dt_convert_float64_to_float32(x.real), dt_convert_float64_to_float32(x.imag));
}

static inline dt_complex128 dt_convert_bool_to_complex128(dt_bool x) {
    return make_complex128(dt_convert_bool_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_uint8_to_complex128(dt_uint8 x) {
    return make_complex128(dt_convert_uint8_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_uint16_to_complex128(dt_uint16 x) {
    return make_complex128(dt_convert_uint16_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_uint32_to_complex128(dt_uint32 x) {
    return make_complex128(dt_convert_uint32_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_uint64_to_complex128(dt_uint64 x) {
    return make_complex128(dt_convert_uint64_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_int8_to_complex128(dt_int8 x) {
    return make_complex128(dt_convert_int8_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_int16_to_complex128(dt_int16 x) {
    return make_complex128(dt_convert_int16_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_int32_to_complex128(dt_int32 x) {
    return make_complex128(dt_convert_int32_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_int64_to_complex128(dt_int64 x) {
    return make_complex128(dt_convert_int64_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_float16_to_complex128(dt_float16 x) {
    return make_complex128(dt_convert_float16_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_bfloat16_to_complex128(dt_bfloat16 x) {
    return make_complex128(dt_convert_bfloat16_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_float32_to_complex128(dt_float32 x) {
    return make_complex128(dt_convert_float32_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_float64_to_complex128(dt_float64 x) {
    return make_complex128(dt_convert_float64_to_float64(x), 0);
}

static inline dt_complex128 dt_convert_complex64_to_complex128(dt_complex64 x) {
    return make_complex128(dt_convert_float32_to_float64(x.real), dt_convert_float32_to_float64(x.imag));
}

static inline dt_complex128 dt_convert_complex128_to_complex128(dt_complex128 x) {
    return x;
}

#endif