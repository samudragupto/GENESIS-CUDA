#include "sort_particles.cuh"
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

void launchSortParticles(int* d_hashes, int* d_indices, int num_particles, cudaStream_t stream) {
    if(num_particles <= 0) return;
    thrust::device_ptr<int> keys(d_hashes);
    thrust::device_ptr<int> values(d_indices);
    thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + num_particles, values);
}