#include "terrain/terrain_normals.cuh"
#include "core/cuda_utils.cuh"
#include "core/constants.cuh"
#include <cstdio>

namespace genesis {

// ═══════════════════════════════════════════════════════
//  KERNEL: Compute normals via central differences
//  Uses shared memory tiling for coalesced reads
// ═══════════════════════════════════════════════════════

#define TILE_DIM 16
#define TILE_PAD 1
#define TILE_FULL (TILE_DIM + 2 * TILE_PAD)

__global__ void k_computeNormals(
    const float* __restrict__ heightmap,
    float* __restrict__ normalX,
    float* __restrict__ normalY,
    float* __restrict__ normalZ,
    int width, int height,
    float heightScale)
{
    __shared__ float tile[TILE_FULL][TILE_FULL];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * TILE_DIM + tx;
    int y = blockIdx.y * TILE_DIM + ty;

    // Load center tile
    int gx = min(max(x, 0), width - 1);
    int gy = min(max(y, 0), height - 1);
    tile[ty + TILE_PAD][tx + TILE_PAD] = heightmap[gy * width + gx];

    // Load halo
    if (tx == 0) {
        int hx = max(x - 1, 0);
        tile[ty + TILE_PAD][0] = heightmap[gy * width + hx];
    }
    if (tx == TILE_DIM - 1 || x == width - 1) {
        int hx = min(x + 1, width - 1);
        tile[ty + TILE_PAD][tx + TILE_PAD + 1] = heightmap[gy * width + hx];
    }
    if (ty == 0) {
        int hy = max(y - 1, 0);
        tile[0][tx + TILE_PAD] = heightmap[hy * width + gx];
    }
    if (ty == TILE_DIM - 1 || y == height - 1) {
        int hy = min(y + 1, height - 1);
        tile[ty + TILE_PAD + 1][tx + TILE_PAD] = heightmap[hy * width + gx];
    }

    __syncthreads();

    if (x >= width || y >= height) return;

    // Central differences
    float hL = tile[ty + TILE_PAD][tx + TILE_PAD - 1];
    float hR = tile[ty + TILE_PAD][tx + TILE_PAD + 1];
    float hD = tile[ty + TILE_PAD - 1][tx + TILE_PAD];
    float hU = tile[ty + TILE_PAD + 1][tx + TILE_PAD];

    float dzdx = (hR - hL) * heightScale * 0.5f;
    float dzdy = (hU - hD) * heightScale * 0.5f;

    // Normal = normalize(cross(tangent_x, tangent_y))
    // tangent_x = (1, 0, dzdx), tangent_y = (0, 1, dzdy)
    // cross = (-dzdx, -dzdy, 1)
    float nx = -dzdx;
    float ny = -dzdy;
    float nz = 1.0f;
    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    nx /= len;
    ny /= len;
    nz /= len;

    int idx = y * width + x;
    normalX[idx] = nx;
    normalY[idx] = ny;
    normalZ[idx] = nz;
}

void TerrainNormals::compute(const float* d_heightmap,
                              float* d_normalX, float* d_normalY, float* d_normalZ,
                              int width, int height, float heightScale,
                              cudaStream_t stream) {
    GpuTimer timer;
    timer.start(stream);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid = gridSize2D(width, height, TILE_DIM);

    k_computeNormals<<<grid, block, 0, stream>>>(
        d_heightmap, d_normalX, d_normalY, d_normalZ,
        width, height, heightScale);
    CUDA_CHECK_LAST();

    timer.stop(stream);
    printf("[TerrainNormals] Computed in %.2f ms\n", timer.elapsedMs());
}

} // namespace genesis