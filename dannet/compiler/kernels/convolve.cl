#ifdef conv1d
__kernel void conv(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC*       C
)
{
    size_t shiftC = get_global_id(0);
    size_t batch = (shiftC / stridesC[0]) % shapeC[0];
    size_t W_out = (shiftC / stridesC[1]) % shapeC[1];
    size_t c_out = (shiftC / stridesC[2]) % shapeC[2];


    W_out *= stride[0];
    dtypeC sum = 0;


    for (size_t kw = 0; kw < shapeB[0]; ++kw)
    {
        for (size_t c_in = 0; c_in < shapeB[1]; ++c_in)
        {
            size_t W_in = W_out + kw;

            dtypeC a = A[
                batch  * stridesA[0] +
                W_in   * stridesA[1] +
                c_in   * stridesA[2] +
                offsetA
            ];
            dtypeB b = B[
                kw     * stridesB[0] +
                c_in   * stridesB[1] +
                c_out  * stridesB[2] +
                offsetB
            ];
            sum += a * b;
        }
    }

    C[shiftC + offsetC] = sum;
}
#endif

#ifdef conv2d
__kernel void conv(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC*       C
)
{
    size_t shiftC = get_global_id(0);
    size_t batch = (shiftC / stridesC[0]) % shapeC[0];
    size_t W_out = (shiftC / stridesC[1]) % shapeC[1];
    size_t H_out = (shiftC / stridesC[2]) % shapeC[2];
    size_t c_out = (shiftC / stridesC[3]) % shapeC[3];


    W_out *= stride[0];
    H_out *= stride[1];
        
    dtypeC sum = 0;


    for (size_t kw = 0; kw < shapeB[0]; ++kw)
    {
        for (size_t kh = 0; kh < shapeB[1]; ++kh)
        {
            for (size_t c_in = 0; c_in < shapeB[2]; ++c_in)
            {
                size_t W_in = W_out + kw;
                size_t H_in = H_out + kh;

                dtypeC a = A[
                    batch  * stridesA[0] +
                    W_in   * stridesA[1] +
                    H_in   * stridesA[2] +
                    c_in   * stridesA[3] +
                    offsetA
                ];
                dtypeB b = B[
                    kw     * stridesB[0] +
                    kh     * stridesB[1] +
                    c_in   * stridesB[2] +
                    c_out  * stridesB[3] +
                    offsetB
                ];
                sum += a * b;
            }
        }
    }

    C[shiftC + offsetC] = sum;
}
#endif

#ifdef conv3d
__kernel void conv(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC*       C
)
{
    size_t shiftC = get_global_id(0);

    size_t batch  = (shiftC / stridesC[0]) % shapeC[0];
    size_t W_out  = (shiftC / stridesC[1]) % shapeC[1];
    size_t H_out  = (shiftC / stridesC[2]) % shapeC[2];
    size_t D_out  = (shiftC / stridesC[3]) % shapeC[3];
    size_t c_out  = (shiftC / stridesC[4]) % shapeC[4];


    W_out *= stride[0];
    H_out *= stride[1];
    D_out *= stride[2];
    
    dtypeC sum = 0;

    for (size_t kw = 0; kw < shapeB[0]; ++kw)
    {
        for (size_t kh = 0; kh < shapeB[1]; ++kh)
        {
            for (size_t kd = 0; kd < shapeB[2]; ++kd)
            {
                for (size_t c_in = 0; c_in < shapeB[3]; ++c_in)
                {
                    size_t W_in = W_out + kw;
                    size_t H_in = H_out + kh;
                    size_t D_in = D_out + kd;

                    dtypeC a = A[
                        batch  * stridesA[0] +
                        W_in   * stridesA[1] +
                        H_in   * stridesA[2] +
                        D_in   * stridesA[3] +
                        c_in   * stridesA[4] +
                        offsetA
                    ];

                    dtypeB b = B[
                        kw     * stridesB[0] +
                        kh     * stridesB[1] +
                        kd     * stridesB[2] +
                        c_in   * stridesB[3] +
                        c_out  * stridesB[4] +
                        offsetB
                    ];

                    sum += a * b;
                }
            }
        }
    }

    C[shiftC + offsetC] = sum;
}
#endif


#ifdef conv2d_depthwise
__kernel void conv(
    __global const dtypeA* A,
    __global const dtypeB* B,
    __global dtypeC*       C
)
{
    size_t shiftC = get_global_id(0);
    size_t batch = (shiftC / stridesC[0]) % shapeC[0];
    size_t W_out = (shiftC / stridesC[1]) % shapeC[1];
    size_t H_out = (shiftC / stridesC[2]) % shapeC[2];
    size_t c_out = (shiftC / stridesC[3]) % shapeC[3];


    W_out *= stride[0];
    H_out *= stride[1];
        
    dtypeC sum = 0;


    for (size_t kw = 0; kw < shapeB[0]; ++kw)
    {
        for (size_t kh = 0; kh < shapeB[1]; ++kh)
        {
            size_t W_in = W_out + kw;
            size_t H_in = H_out + kh;

            dtypeC a = A[
                batch  * stridesA[0] +
                W_in   * stridesA[1] +
                H_in   * stridesA[2] +
                c_out  * stridesA[3] +
                offsetA
            ];
            dtypeB b = B[
                kw     * stridesB[0] +
                kh     * stridesB[1] +
                c_out  * stridesB[2] +
                offsetB
            ];
            sum += a * b;
        }
    }

    C[shiftC + offsetC] = sum;
}
#endif