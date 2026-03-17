#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "hydraulic_erosion.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <curand_kernel.h>

__global__ void hydraulicErosionKernel(
    float* __restrict__ heightmap,
    int world_size,
    int num_droplets,
    unsigned int seed,
    int max_lifetime
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_droplets) return;

    curandState local_rng;
    curand_init(seed, idx, 0, &local_rng);

    float posX = curand_uniform(&local_rng) * (world_size - 1);
    float posY = curand_uniform(&local_rng) * (world_size - 1);
    
    float dirX = 0.0f;
    float dirY = 0.0f;
    float speed = 1.0f;
    float water = 1.0f;
    float sediment = 0.0f;

    for (int lifetime = 0; lifetime < max_lifetime; lifetime++) {
        int nodeX = (int)posX;
        int nodeY = (int)posY;
        int dropletIndex = nodeY * world_size + nodeX;
        
        float cellOffsetX = posX - nodeX;
        float cellOffsetY = posY - nodeY;

        float heightNW = heightmap[dropletIndex];
        float heightNE = heightmap[dropletIndex + 1];
        float heightSW = heightmap[dropletIndex + world_size];
        float heightSE = heightmap[dropletIndex + world_size + 1];

        float gradientX = (heightNE - heightNW) * (1.0f - cellOffsetY) + (heightSE - heightSW) * cellOffsetY;
        float gradientY = (heightSW - heightNW) * (1.0f - cellOffsetX) + (heightSE - heightNE) * cellOffsetX;

        dirX = (dirX * EROSION_INERTIA) - gradientX * (1.0f - EROSION_INERTIA);
        dirY = (dirY * EROSION_INERTIA) - gradientY * (1.0f - EROSION_INERTIA);

        float len = sqrtf(dirX * dirX + dirY * dirY);
        if (len != 0) {
            dirX /= len;
            dirY /= len;
        }

        posX += dirX;
        posY += dirY;

        if (posX < 0 || posX >= world_size - 1 || posY < 0 || posY >= world_size - 1) break;

        float newHeightNW = heightmap[(int)posY * world_size + (int)posX];
        float newHeightNE = heightmap[(int)posY * world_size + (int)posX + 1];
        float newHeightSW = heightmap[((int)posY + 1) * world_size + (int)posX];
        float newHeightSE = heightmap[((int)posY + 1) * world_size + (int)posX + 1];
        
        float newHeight = newHeightNW * (1.0f - cellOffsetX) * (1.0f - cellOffsetY) +
                          newHeightNE * cellOffsetX * (1.0f - cellOffsetY) +
                          newHeightSW * (1.0f - cellOffsetX) * cellOffsetY +
                          newHeightSE * cellOffsetX * cellOffsetY;

        float deltaHeight = newHeight - (heightNW * (1.0f - cellOffsetX) * (1.0f - cellOffsetY) +
                                         heightNE * cellOffsetX * (1.0f - cellOffsetY) +
                                         heightSW * (1.0f - cellOffsetX) * cellOffsetY +
                                         heightSE * cellOffsetX * cellOffsetY);

        float sedimentCapacity = fmaxf(-deltaHeight * speed * water * EROSION_SEDIMENT_CAP, 0.01f);
        
        if (sediment > sedimentCapacity || deltaHeight > 0) {
            float amountToDeposit = (deltaHeight > 0) ? fminf(deltaHeight, sediment) : (sediment - sedimentCapacity) * EROSION_DEPOSIT_SPEED;
            sediment -= amountToDeposit;

            atomicAdd(&heightmap[dropletIndex], amountToDeposit * 0.25f);
            atomicAdd(&heightmap[dropletIndex + 1], amountToDeposit * 0.25f);
            atomicAdd(&heightmap[dropletIndex + world_size], amountToDeposit * 0.25f);
            atomicAdd(&heightmap[dropletIndex + world_size + 1], amountToDeposit * 0.25f);
        } else {
            float amountToErode = fminf((sedimentCapacity - sediment) * EROSION_ERODE_SPEED, -deltaHeight);
            sediment += amountToErode;

            atomicAdd(&heightmap[dropletIndex], -amountToErode * 0.25f);
            atomicAdd(&heightmap[dropletIndex + 1], -amountToErode * 0.25f);
            atomicAdd(&heightmap[dropletIndex + world_size], -amountToErode * 0.25f);
            atomicAdd(&heightmap[dropletIndex + world_size + 1], -amountToErode * 0.25f);
        }

        speed = sqrtf(speed * speed + deltaHeight * GRAVITY);
        water *= (1.0f - EROSION_EVAPORATE_SPEED);
    }
}

void launchHydraulicErosion(
    float* d_heightmap,
    int world_size,
    int num_droplets,
    unsigned int seed,
    int max_lifetime,
    cudaStream_t stream
) {
    if (num_droplets <= 0) return;
    int block = 256;
    // THIS LINE WAS WRONG: int grid = computeGrid1D(num_droplets, block);
    // REPLACE IT WITH THIS:
    int grid = (num_droplets + block - 1) / block;
    
    hydraulicErosionKernel<<<grid, block, 0, stream>>>(
        d_heightmap, world_size, num_droplets, seed, max_lifetime
    );
    KERNEL_CHECK();
}