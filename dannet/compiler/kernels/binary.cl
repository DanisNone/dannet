__kernel void binary(
    __global const dt_$dtypeA$ *A,
    __global const dt_$dtypeB$* B,
    __global dt_$dtypeC$ *C, 
    ShapeInfo Ainfo,
    ShapeInfo Binfo,
    ShapeInfo Cinfo
)
{
    size_t x = get_global_id(0);
    // C is default strided 
    
    size_t Aoffset = Ainfo.buffer_offset;
    size_t Boffset = Binfo.buffer_offset;
    for (int i = 0; i < Cinfo.ndim; i++) {
        size_t index = x / Cinfo.strides[i];
        Aoffset += (index % Ainfo.shape[i]) * Ainfo.strides[i];
        Boffset += (index % Binfo.shape[i]) * Binfo.strides[i];
    }

    C[x] = operation(A[Aoffset], B[Boffset]);
}