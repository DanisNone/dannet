__kernel void general(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC* C
)
{
    size_t batch_shift = get_global_id(0) * M * K;
    size_t x = get_global_id(1);
    size_t z = get_global_id(2);

    size_t lx = get_local_id(1);
    size_t lz = get_local_id(2);
    
    size_t Ashift = 0;
    size_t Bshift = 0;
    
    for (int axis = 0; axis < ndimC - 2; axis++)
    {
        Ashift += ((batch_shift / stridesC[axis]) % shapeA[axis]) * stridesA[axis];
        Bshift += ((batch_shift / stridesC[axis]) % shapeB[axis]) * stridesB[axis];
    }

    __local dtypeC Asub[tile_size][tile_size];
    __local dtypeC Bsub[tile_size][tile_size];
    
    dtypeC res = 0;
    for (size_t y = 0; y < N; y += tile_size)
    {
        if (x < M && (y + lz) < N)
            Asub[lx][lz] = A[Ashift + x * stridesA[ndimA - 2] + (y + lz) * stridesA[ndimA - 1]];
        else
            Asub[lx][lz] = 0;
        
        if ((y + lx) < N && z < K)
            Bsub[lx][lz] = B[Bshift + (y + lx) * stridesB[ndimB - 2] + z * stridesB[ndimB - 1]];
        else
            Bsub[lx][lz] = 0;
            
        barrier(CLK_LOCAL_MEM_FENCE);
        
        
        for (size_t ly = 0; ly < tile_size; ly++)
            res += Asub[lx][ly] * Bsub[ly][lz];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < M && z < K)
        C[batch_shift + x * K + z] = res;
}
