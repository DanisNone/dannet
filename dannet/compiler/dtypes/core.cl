#ifndef _CORE_CL_
#define _CORE_CL_

#include "dtypes/dtypes.cl"
#include "dtypes/convert.cl"

typedef struct {
    size_t buffer_offset;
    size_t ndim;
    size_t shape[64];
    size_t strides[64];
} ShapeInfo;

#endif