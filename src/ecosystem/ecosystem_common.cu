#include "ecosystem_common.cuh"
#include "../core/cuda_utils.cuh"

void allocateEcosystemGrid(EcosystemGridData& grid, int world_size) {
    grid.world_size = world_size;
    grid.total_cells = world_size * world_size;

    CUDA_CHECK(cudaMalloc(&grid.d_vegetation, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid.d_vegetation_capacity, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid.d_dead_matter, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid.d_mineral_nutrients, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grid.d_toxicity, grid.total_cells * sizeof(float)));

    CUDA_CHECK(cudaMemset(grid.d_vegetation, 0, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grid.d_vegetation_capacity, 0, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grid.d_dead_matter, 0, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grid.d_mineral_nutrients, 0, grid.total_cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grid.d_toxicity, 0, grid.total_cells * sizeof(float)));
}

void freeEcosystemGrid(EcosystemGridData& grid) {
    cudaFree(grid.d_vegetation);
    cudaFree(grid.d_vegetation_capacity);
    cudaFree(grid.d_dead_matter);
    cudaFree(grid.d_mineral_nutrients);
    cudaFree(grid.d_toxicity);
}

void allocatePopulationStats(PopulationStats& stats, int max_species) {
    stats.max_species = max_species;
    CUDA_CHECK(cudaMalloc(&stats.d_species_pop_count, max_species * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&stats.d_species_avg_energy, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&stats.d_species_avg_fitness, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&stats.d_total_alive, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&stats.d_total_births, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&stats.d_total_deaths, sizeof(int)));

    CUDA_CHECK(cudaMemset(stats.d_species_pop_count, 0, max_species * sizeof(int)));
    CUDA_CHECK(cudaMemset(stats.d_species_avg_energy, 0, max_species * sizeof(float)));
    CUDA_CHECK(cudaMemset(stats.d_species_avg_fitness, 0, max_species * sizeof(float)));
    CUDA_CHECK(cudaMemset(stats.d_total_alive, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(stats.d_total_births, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(stats.d_total_deaths, 0, sizeof(int)));
}

void freePopulationStats(PopulationStats& stats) {
    cudaFree(stats.d_species_pop_count);
    cudaFree(stats.d_species_avg_energy);
    cudaFree(stats.d_species_avg_fitness);
    cudaFree(stats.d_total_alive);
    cudaFree(stats.d_total_births);
    cudaFree(stats.d_total_deaths);
}