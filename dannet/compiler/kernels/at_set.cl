__kernel void at_set(
    __global const dt_$dtypeA$ *A,
    __global dt_$dtypeB$ *B, 
    ShapeInfo Ainfo, 
    ShapeInfo Binfo,
    Shape start,
    Shape step,
    Shape norm_strides
)
{
    size_t x = get_global_id(0);
    // B is default strided 
    
    
    size_t Aoffset = Ainfo.buffer_offset;
    size_t Boffset = 0;
    for (int i = 0; i < Ainfo.ndim; i++) {
        size_t idx = (x / norm_strides.data[i]) % Ainfo.shape[i];
        Aoffset += idx * Ainfo.strides[i];
        Boffset += (idx * step.data[i] + start.data[i]) * Binfo.strides[i];
    }

    B[Boffset] = dt_convert_$dtypeA$_to_$dtypeB$(A[Aoffset]);
}