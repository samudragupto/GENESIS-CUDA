#ifndef SORT_PARTICLES_CUH
#define SORT_PARTICLES_CUH

#include <cuda_runtime.h>

void launchSortParticles(int* d_hashes, int* d_indices, int num_particles, cudaStream_t stream = 0);

#endif