#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "genome.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__constant__ int d_gene_permutation[512]; // Note: size must be 512 for Perlin Noise!

void initGenomeConstants() {
    int h_perm[512];
    // Standard Perlin noise permutation table initialization
    for (int i = 0; i < 256; i++) {
        h_perm[i] = i;
    }
    
    // Shuffle
    srand(42);
    for (int i = 255; i > 0; i--) {
        int j = rand() % (i + 1);
        int tmp = h_perm[i];
        h_perm[i] = h_perm[j];
        h_perm[j] = tmp;
    }
    
    // Duplicate to avoid buffer overflows in Perlin noise calculation (p[x] + y)
    for (int i = 0; i < 256; i++) {
        h_perm[256 + i] = h_perm[i];
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_gene_permutation, h_perm, 512 * sizeof(int)));
}