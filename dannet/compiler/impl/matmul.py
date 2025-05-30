import math
import dannet as dt

from .utils import (
    Generator,
    build_kernel,
    register_impl
)


@register_impl(dt.math._Matmul)
def matmul(
    device,
    node: dt.math._Matmul,
    input_buffers,
    output_buffer,
):
    A, B = input_buffers
    C = output_buffer
    assert node._is_contiguous

    shapeA, shapeB = node.x._shape, node.y._shape
    shapeC = node._shape

    assert len(shapeA) >= 2 and len(shapeB) >= 2

    shapeC = dt.utils.broadcast_shapes(
        shapeA[:-2], shapeB[:-2]
    ) + (shapeA[-2], shapeB[-1])
    shapeA = dt.utils.broadcast_shape_to(
        shapeA[:-2], shapeC[:-2]
    ) + shapeA[-2:]
    shapeB = dt.utils.broadcast_shape_to(
        shapeB[:-2], shapeC[:-2]
    ) + shapeB[-2:]

    M, N = shapeA[-2:]
    N0, K = shapeB[-2:]
    M0, K0 = shapeC[-2:]

    assert M == M0 and N == N0 and K == K0

    tile_size = min(max(M, K), math.isqrt(device.max_work_group_size))
    global_size = (
        (M + tile_size - 1) // tile_size * tile_size,
        (K + tile_size - 1) // tile_size * tile_size,
    )
    local_size = (tile_size, tile_size)

    gen = Generator()
    gen.nodes_info(
        A=node.x,
        B=node.y,
        C=node
    )
    gen.defines(M=M, N=N, K=K, tile_size=tile_size)

    kernel = build_kernel(device, 'matmul.cl', gen)
    return lambda: kernel.general(
        device.queue, global_size, local_size,
        A, B, C
    )
