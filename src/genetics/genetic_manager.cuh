#ifndef GENETIC_MANAGER_CUH
#define GENETIC_MANAGER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "../core/constants.cuh"
#include "genome.cuh"
#include "crossover_kernels.cuh"
#include "mutation_kernels.cuh"
#include "speciation.cuh"
#include "phylogenetic_tree.cuh"
#include "fitness_evaluation.cuh"

class GeneticManager {
public:
    curandState* d_rng_states;
    float* d_fitness;
    int* d_eligible;
    int* d_eligible_count;
    int* d_selected;
    ReproductionPair* d_repro_pairs;
    float* d_child_genomes;
    int* d_species_counter;

    MutationParams mutation_params;
    int max_creatures;
    int num_rng_states;

    cudaStream_t stream_genetics;

    void init(int max_c);
    void destroy();

    void evaluateFitness(const float* d_genomes, const float* d_energy,
        const int* d_age, const int* d_alive, int num_creatures);

    int reproduce(float* d_genomes, int* d_species_id, int* d_alive,
        const float* d_energy, const int* d_age, int num_alive,
        int max_offspring, float speciation_threshold);
};

#endif