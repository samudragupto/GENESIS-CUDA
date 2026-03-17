#include "creature_common.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateCreatureData(CreatureData& data, int max_creatures) {
    data.max_creatures = max_creatures;
    CUDA_CHECK(cudaMalloc(&data.d_pos_x, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_pos_y, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_vel_x, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_vel_y, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_energy, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_health, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_age, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_species_id, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_state, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_repro_cooldown, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_genomes, (size_t)max_creatures * GENOME_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_neural_weights, (size_t)max_creatures * NEURAL_WEIGHT_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_alive, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_num_alive, sizeof(int)));

    CUDA_CHECK(cudaMemset(data.d_pos_x, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_pos_y, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_vel_x, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_vel_y, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_energy, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_health, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_age, 0, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMemset(data.d_species_id, 0, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMemset(data.d_state, 0, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMemset(data.d_repro_cooldown, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_genomes, 0, (size_t)max_creatures * GENOME_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_neural_weights, 0, (size_t)max_creatures * NEURAL_WEIGHT_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_alive, 0, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMemset(data.d_num_alive, 0, sizeof(int)));
}

void freeCreatureData(CreatureData& data) {
    cudaFree(data.d_pos_x);
    cudaFree(data.d_pos_y);
    cudaFree(data.d_vel_x);
    cudaFree(data.d_vel_y);
    cudaFree(data.d_energy);
    cudaFree(data.d_health);
    cudaFree(data.d_age);
    cudaFree(data.d_species_id);
    cudaFree(data.d_state);
    cudaFree(data.d_repro_cooldown);
    cudaFree(data.d_genomes);
    cudaFree(data.d_neural_weights);
    cudaFree(data.d_alive);
    cudaFree(data.d_num_alive);
}

void allocateInteractionBuffers(InteractionBuffers& buf, int max_creatures) {
    buf.max_creatures = max_creatures;
    CUDA_CHECK(cudaMalloc(&buf.d_reproduce_flag, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_mate_index, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_mating_lock, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_spawn_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_new_creature_indices, max_creatures * sizeof(int)));
}

void freeInteractionBuffers(InteractionBuffers& buf) {
    cudaFree(buf.d_reproduce_flag);
    cudaFree(buf.d_mate_index);
    cudaFree(buf.d_mating_lock);
    cudaFree(buf.d_spawn_counter);
    cudaFree(buf.d_new_creature_indices);
}