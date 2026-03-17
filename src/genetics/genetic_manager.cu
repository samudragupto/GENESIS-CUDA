#include "genetic_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include "../genetics/curand_states.cuh"

__global__ void initRandomGenomesKernel(
    float* __restrict__ genomes,
    curandState* __restrict__ rng,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    curandState local_rng = rng[idx];
    int base = idx * GENOME_LENGTH;

    for (int g = 0; g < GENOME_LENGTH; g++) {
        genomes[base + g] = curand_uniform(&local_rng);
    }

    genomes[base + GENE_MUTATION_RATE] = 0.3f + curand_uniform(&local_rng) * 0.4f;
    genomes[base + GENE_MUTATION_STRENGTH] = 0.3f + curand_uniform(&local_rng) * 0.4f;

    rng[idx] = local_rng;
}

__global__ void buildReproPairsKernel(
    ReproductionPair* __restrict__ pairs,
    const int* __restrict__ eligible_indices,
    const int* __restrict__ selected_mates,
    int num_pairs,
    int child_start_index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    pairs[idx].parent_a = eligible_indices[idx];
    pairs[idx].parent_b = selected_mates[idx];
    pairs[idx].child_index = child_start_index + idx;
}

void GeneticManager::init(int max_c) {
    max_creatures = max_c;
    num_rng_states = max_c;

    CUDA_CHECK(cudaMalloc(&d_rng_states, max_c * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_fitness, max_c * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eligible, max_c * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eligible_count, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_selected, max_c * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_repro_pairs, (max_c / 10) * sizeof(ReproductionPair)));
    CUDA_CHECK(cudaMalloc(&d_child_genomes, (size_t)(max_c / 10) * GENOME_LENGTH * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_species_counter, sizeof(int)));

    int init_species = 10;
    CUDA_CHECK(cudaMemcpy(d_species_counter, &init_species, sizeof(int), cudaMemcpyHostToDevice));

    launchInitRNG(d_rng_states, max_c, 42ULL, 0);

    mutation_params.base_rate = 0.02f;
    mutation_params.gaussian_sigma = 0.1f;
    mutation_params.reset_prob = 0.01f;
    mutation_params.duplication_prob = 0.005f;
    mutation_params.deletion_prob = 0.005f;

    CUDA_CHECK(cudaStreamCreate(&stream_genetics));
}

void GeneticManager::destroy() {
    cudaFree(d_rng_states);
    cudaFree(d_fitness);
    cudaFree(d_eligible);
    cudaFree(d_eligible_count);
    cudaFree(d_selected);
    cudaFree(d_repro_pairs);
    cudaFree(d_child_genomes);
    cudaFree(d_species_counter);
    cudaStreamDestroy(stream_genetics);
}

void GeneticManager::evaluateFitness(const float* d_genomes, const float* d_energy,
    const int* d_age, const int* d_alive, int num_creatures) {
    launchFitnessEvaluation(d_genomes, d_energy, d_age, d_alive, d_fitness, num_creatures, stream_genetics);
}

int GeneticManager::reproduce(float* d_genomes, int* d_species_id, int* d_alive,
    const float* d_energy, const int* d_age, int num_alive,
    int max_offspring, float speciation_threshold) {
    (void)d_species_id;
    (void)d_alive;
    (void)speciation_threshold;

    launchEligibilityCheck(d_energy, d_age, d_alive, d_genomes,
        d_eligible, d_eligible_count, num_alive, stream_genetics);
    CUDA_CHECK(cudaStreamSynchronize(stream_genetics));

    int h_eligible_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_eligible_count, d_eligible_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_eligible_count <= 1) return 0;

    int num_offspring = (h_eligible_count / 2 < max_offspring) ? h_eligible_count / 2 : max_offspring;
    if (num_offspring <= 0) return 0;

    launchTournamentSelection(d_fitness, d_alive, d_selected,
        num_alive, num_offspring, 3, d_rng_states, stream_genetics);

    CUDA_CHECK(cudaStreamSynchronize(stream_genetics));
    return num_offspring;
}