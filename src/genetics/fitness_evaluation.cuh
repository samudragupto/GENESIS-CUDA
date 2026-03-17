#ifndef FITNESS_EVALUATION_CUH
#define FITNESS_EVALUATION_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../core/constants.cuh"

void launchFitnessEvaluation(
    const float* d_genomes,
    const float* d_energy,
    const int* d_age,
    const int* d_alive,
    float* d_fitness,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchTournamentSelection(
    const float* d_fitness,
    const int* d_alive,
    int* d_selected,
    int num_creatures,
    int num_to_select,
    int tournament_size,
    curandState* rng_states,
    cudaStream_t stream = 0
);

void launchEligibilityCheck(
    const float* d_energy,
    const int* d_age,
    const int* d_alive,
    const float* d_genomes,
    int* d_eligible,
    int* d_eligible_count,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif