__kernel void binary(
    __global const dt_$dtypeA$ *A,
    __global const dt_$dtypeB$* B,
    __global const dt_$dtypeC$* C,
    __global dt_$dtypeD$ *D, 
    ShapeInfo Ainfo,
    ShapeInfo Binfo,
    ShapeInfo Cinfo,
    ShapeInfo Dinfo
)
{
    size_t x = get_global_id(0);
    // D is default strided 
    
    size_t Aoffset = Ainfo.buffer_offset;
    size_t Boffset = Binfo.buffer_offset;
    size_t Coffset = Cinfo.buffer_offset;
    for (int i = 0; i < Dinfo.ndim; i++) {
        size_t index = x / Dinfo.strides[i];
        Aoffset += (index % Ainfo.shape[i]) * Ainfo.strides[i];
        Boffset += (index % Binfo.shape[i]) * Binfo.strides[i];
        Coffset += (index % Cinfo.shape[i]) * Cinfo.strides[i];
    }

    D[x] = operation(A[Aoffset], B[Boffset], C[Coffset]);
}