#ifndef CURAND_STATES_CUH
#define CURAND_STATES_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

void launchInitRNG(curandState* states, int count, unsigned long long seed, cudaStream_t stream = 0);

#endif