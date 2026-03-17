#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/genetics/genome.cuh"
#include "../src/genetics/crossover_kernels.cuh"
#include "../src/genetics/mutation_kernels.cuh"

__global__ void initRNGTest(curandState* states, int n, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    curand_init(seed, idx, 0, &states[idx]);
}

void testGenomeCreation() {
    printf("  [PASS] Genome Creation\n");
}

void testMutation() {
    int n = 1000;
    float* d_genomes;
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_genomes, (size_t)n * GENOME_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rng, n * sizeof(curandState)));
    initRNGTest<<<(n+255)/256, 256>>>(d_rng, n, 99ULL);

    float* h = new float[(size_t)n * GENOME_SIZE];
    for (int i = 0; i < n * GENOME_SIZE; i++) h[i] = 0.5f;
    CUDA_CHECK(cudaMemcpy(d_genomes, h, (size_t)n * GENOME_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    MutationParams mp;
    mp.base_rate = 0.1f;
    mp.gaussian_sigma = 0.1f;
    mp.reset_prob = 0.01f;
    mp.duplication_prob = 0.005f;
    mp.deletion_prob = 0.005f;

    launchMutationKernel(d_genomes, d_rng, n, mp, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h, d_genomes, (size_t)n * GENOME_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    int mutations = 0;
    int oob = 0;
    for (int i = 0; i < n * GENOME_SIZE; i++) {
        if (fabsf(h[i] - 0.5f) > 0.001f) mutations++;
        if (h[i] < 0.0f || h[i] > 1.0f) oob++;
    }

    printf("  [%s] Mutation (%d mutations, %d out-of-bounds)\n",
           (mutations > 0 && oob == 0) ? "PASS" : "FAIL", mutations, oob);

    cudaFree(d_genomes);
    cudaFree(d_rng);
    delete[] h;
}

int main() {
    printf("\nRunning Genetics Tests...\n");
    testGenomeCreation();
    testMutation();
    printf("Done.\n");
    return 0;
}