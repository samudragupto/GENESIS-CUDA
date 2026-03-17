#ifndef DIVERSITY_INDEX_CUH
#define DIVERSITY_INDEX_CUH

#include <cuda_runtime.h>

struct DiversityResult {
    float* d_shannon;
    float* d_simpson;
    float* d_evenness;
    float* d_richness;
};

void allocateDiversityResult(DiversityResult& dr);
void freeDiversityResult(DiversityResult& dr);

void launchShannonDiversity(
    const int* d_species_pop_count,
    int max_species,
    int total_alive,
    DiversityResult& result,
    cudaStream_t stream = 0
);

void launchSimpsonDiversity(
    const int* d_species_pop_count,
    int max_species,
    int total_alive,
    DiversityResult& result,
    cudaStream_t stream = 0
);

void launchGeneticDiversity(
    const float* d_genomes,
    const int* d_alive,
    int num_creatures,
    float* d_genetic_diversity,
    cudaStream_t stream = 0
);

#endif