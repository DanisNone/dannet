__kernel void matmul(
    __global const dt_$dtypeA$ *A,
    __global const dt_$dtypeB$ *B,
    __global dt_$dtypeC$ *C, 
    ShapeInfo Ainfo,
    ShapeInfo Binfo,
    ShapeInfo Cinfo
)
{
    size_t batch = get_global_id(0);
    size_t x = get_global_id(1);
    size_t z = get_global_id(2);
    size_t ndim = Cinfo.ndim;
    // C is default strided 
    
    size_t Aoffset = Ainfo.buffer_offset + x * Ainfo.strides[ndim - 2];
    size_t Boffset = Binfo.buffer_offset + z * Binfo.strides[ndim - 1];

    batch *= Cinfo.shape[ndim - 1] * Cinfo.shape[ndim - 2];
    for (int i = 0; i < ndim - 2; i++) {
        size_t index = batch / Cinfo.strides[i];
        Aoffset += (index % Ainfo.shape[i]) * Ainfo.strides[i];
        Boffset += (index % Binfo.shape[i]) * Binfo.strides[i];
    }

    size_t N = Ainfo.shape[ndim - 1];
    dt_$dtypeC$ result = dt_zero_$dtypeC$();
    for (size_t y = 0; y < N; y++)
    {
        dt_$dtypeC$ a = dt_convert_$dtypeA$_to_$dtypeC$(A[Aoffset + y * Ainfo.strides[ndim - 1]]);
        dt_$dtypeC$ b = dt_convert_$dtypeB$_to_$dtypeC$(B[Boffset + y * Binfo.strides[ndim - 2]]);

        result = dt_mad_$dtypeC$(a, b, result);
    }
    C[batch + x * Cinfo.shape[ndim - 1] + z] = result;
}