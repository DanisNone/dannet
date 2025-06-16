__kernel void reduction(
    __global const dt_$dtypeA$ *A,
    __global dt_$dtypeB$ *B, 
    ShapeInfo Ainfo, 
    ShapeInfo Binfo, 
    Shape inner_strides,
    size_t inner_size 
)
{
    size_t x = get_global_id(0);
    // B is default strided 
    
    size_t Aoffset = Ainfo.buffer_offset;
    for (int i = 0; i < Binfo.ndim; i++) {
        Aoffset += ((x / Binfo.strides[i]) % Ainfo.shape[i]) * Ainfo.strides[i];
    }

    dt_$dtypeB$ acc = init(A[Aoffset]);
    for (size_t i = 1; i < inner_size; i++)
    {
        size_t Ainner_offset = Aoffset;
        for (int axis = 0; axis < inner_strides.ndim; axis++)
        {
            Ainner_offset += ((i / inner_strides.data[axis]) % Ainfo.shape[axis + Binfo.ndim]) * Ainfo.strides[axis + Binfo.ndim];
        }
        acc = operation(acc, A[Ainner_offset]);
    }

    B[x] = final(acc, inner_size);
}