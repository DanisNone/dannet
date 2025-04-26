__kernel void general(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC* C
)
{
    long batch_shift = get_global_id(0) * M * K;
    long x = get_global_id(1);
    long z = get_global_id(2);

    long lx = get_local_id(1);
    long lz = get_local_id(2);
    
    long Ashift = 0;
    long Bshift = 0;
    
    for (int axis = 0; axis < ndimC - 2; axis++)
    {
        Ashift += ((batch_shift / stridesC[axis]) % shapeA[axis]) * stridesA[axis];
        Bshift += ((batch_shift / stridesC[axis]) % shapeB[axis]) * stridesB[axis];
    }

    __local dtypeC Asub[tile_size][tile_size];
    __local dtypeC Bsub[tile_size][tile_size];
    
    dtypeC res = 0;
    for (long y = 0; y < N; y += tile_size)
    {
        if (x < M && (y + lz) < N)
            Asub[lx][lz] = A[Ashift + x * N + (y + lz)];
        else
            Asub[lx][lz] = 0;
        
        if ((y + lx) < N && z < K)
            Bsub[lx][lz] = B[Bshift + (y + lx) * K + z];
        else
            Bsub[lx][lz] = 0;
            
        barrier(CLK_LOCAL_MEM_FENCE);
        
        
        for (long ly = 0; ly < tile_size; ly++)
            res += Asub[lx][ly] * Bsub[ly][lz];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < M && z < K)
        C[batch_shift + x * K + z] = res;
}
