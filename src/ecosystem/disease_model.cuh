#ifndef DISEASE_MODEL_CUH
#define DISEASE_MODEL_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../creatures/creature_common.cuh"

enum DiseaseState : int {
    DISEASE_SUSCEPTIBLE = 0,
    DISEASE_INFECTED = 1,
    DISEASE_RECOVERED = 2,
    DISEASE_DEAD = 3
};

struct DiseaseData {
    int*   d_disease_state;
    float* d_infection_timer;
    float* d_immunity;
    int    max_creatures;
};

struct DiseaseParams {
    float transmission_radius;
    float transmission_prob;
    float infection_duration;
    float mortality_rate;
    float immunity_duration;
    float health_damage_rate;
    float energy_drain_rate;
    float mutation_chance;
    float dt;
};

void allocateDiseaseData(DiseaseData& data, int max_creatures);
void freeDiseaseData(DiseaseData& data);

void launchDiseaseSpread(
    DiseaseData& disease,
    const CreatureData& creatures,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const DiseaseParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream = 0
);

void launchDiseaseProgress(
    DiseaseData& disease,
    CreatureData& creatures,
    const DiseaseParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream = 0
);

void launchDiseaseSeed(
    DiseaseData& disease,
    int num_creatures,
    int num_to_infect,
    curandState* d_rng,
    cudaStream_t stream = 0
);

#endif