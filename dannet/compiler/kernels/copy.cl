__kernel void copy(
    __global const dt_$dtypeA$ *A,
    __global dt_$dtypeA$ *B, 
    ShapeInfo Ainfo, 
    ShapeInfo Binfo
)
{
    size_t x = get_global_id(0);
    // B is default strided 
    
    size_t Aoffset = Ainfo.buffer_offset;
    for (int i = 0; i < Ainfo.ndim; i++) {
        Aoffset += ((x / Binfo.strides[i]) % Ainfo.shape[i]) * Ainfo.strides[i];
    }

    B[x] = A[Aoffset];
}