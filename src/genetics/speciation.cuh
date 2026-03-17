#pragma once

#include "genome.cuh"
#include <cuda_runtime.h>

namespace genesis {
namespace genetics {

struct SpeciationParams {
    float distance_threshold;
    int comparison_genes;
    int min_population_for_species;
    float centroid_update_rate;
};

void launchSpeciationCheckKernel(
    const float* genomes,
    int* species_ids,
    const float* species_centroids,
    const int* parent_species,
    int num_children,
    const int* child_indices,
    int* d_next_species_id,
    int* d_new_species_flags,
    const SpeciationParams& params,
    cudaStream_t stream = 0);

void launchCentroidAccumulateKernel(
    const float* genomes,
    const int* species_ids,
    float* centroid_accum,
    int* species_counts,
    int num_creatures,
    int num_species,
    cudaStream_t stream = 0);

void launchCentroidNormalizeKernel(
    float* centroids,
    const float* centroid_accum,
    const int* species_counts,
    int num_species,
    float update_rate,
    cudaStream_t stream = 0);

void launchPopulationCountKernel(
    const int* species_ids,
    int* species_populations,
    int num_creatures,
    int num_species,
    cudaStream_t stream = 0);

}
}