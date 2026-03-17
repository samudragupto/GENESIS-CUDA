#include "curand_states.cuh"
#include "../core/cuda_utils.cuh"

__global__ void initRNGKernel(curandState* states, int count, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    curand_init(seed, idx, 0, &states[idx]);
}

void launchInitRNG(curandState* states, int count, unsigned long long seed, cudaStream_t stream) {
    if (count <= 0) return;
    int block = 256;
    int grid = (count + block - 1) / block;
    initRNGKernel<<<grid, block, 0, stream>>>(states, count, seed);
}