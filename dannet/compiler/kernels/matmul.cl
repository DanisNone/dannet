__kernel void general(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC* C
)
{
    size_t x = get_global_id(0);
    size_t z = get_global_id(1);

    size_t lx = get_local_id(0);
    size_t lz = get_local_id(1);
    

    __local dtypeC subA[tile_size][tile_size];
    __local dtypeC subB[tile_size][tile_size];
    

    for (size_t batch = 0; batch < sizeC / (M * K); batch++)
    {
        size_t batch_shift = batch * M * K;
        size_t shiftA = 0;
        size_t shiftB = 0;
        
        for (int axis = 0; axis < ndimC - 2; axis++)
        {
            shiftA += ((batch_shift / stridesC[axis]) % shapeA[axis]) * stridesA[axis];
            shiftB += ((batch_shift / stridesC[axis]) % shapeB[axis]) * stridesB[axis];
        }

        dtypeC res = dt_zero_dtypeC();
        for (size_t y = 0; y < N; y += tile_size)
        {
            dtypeA a;
            dtypeB b;

            if (x < M && (y + lz) < N)
                a = A[shiftA + x * stridesA[ndimA - 2] + (y + lz) * stridesA[ndimA - 1]];
            else
                a = dt_zero_dtypeA();
            
            if ((y + lx) < N && z < K)
                b = B[shiftB + (y + lx) * stridesB[ndimB - 2] + z * stridesB[ndimB - 1]];
            else
                b = dt_zero_dtypeB();
                
            subA[lx][lz] = dt_convert_dtypeA_to_dtypeC(a);
            subB[lz][lx] = dt_convert_dtypeB_to_dtypeC(b);
                
            barrier(CLK_LOCAL_MEM_FENCE);
            
            
            for (size_t ly = 0; ly < tile_size; ly++)
                res = dt_mad_dtypeC(subA[lx][ly], subB[lz][ly], res);
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (x < M && z < K)
            C[batch_shift + x * K + z] = res;
    }
}
