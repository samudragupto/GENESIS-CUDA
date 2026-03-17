#include "speciation.cuh"
#include "../core/cuda_utils.cuh"
#include "speciation.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
namespace genesis {
namespace genetics {

__global__ void speciationCheckKernel(
    const float* __restrict__ genomes,
    int* __restrict__ species_ids,
    const float* __restrict__ species_centroids,
    const int* __restrict__ parent_species,
    int num_children,
    const int* __restrict__ child_indices,
    int* __restrict__ d_next_species_id,
    int* __restrict__ d_new_species_flags,
    float distance_threshold,
    int comparison_genes)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_children) return;

    int child = child_indices[idx];
    int parent_sp = parent_species[idx];

    int centroid_base = parent_sp * CENTROID_GENES;
    int child_base = child * GENOME_LENGTH;

    float dist = 0.0f;
    int genes_to_compare = min(comparison_genes, CENTROID_GENES);

    for (int g = 0; g < genes_to_compare; g++) {
        float diff = genomes[child_base + g] - species_centroids[centroid_base + g];
        dist += diff * diff;
    }
    dist = sqrtf(dist / (float)genes_to_compare);

    if (dist > distance_threshold) {
        int new_id = atomicAdd(d_next_species_id, 1);
        species_ids[child] = new_id;
        d_new_species_flags[idx] = 1;
    } else {
        species_ids[child] = parent_sp;
        d_new_species_flags[idx] = 0;
    }
}

__global__ void centroidAccumulateKernel(
    const float* __restrict__ genomes,
    const int* __restrict__ species_ids,
    float* __restrict__ centroid_accum,
    int* __restrict__ species_counts,
    int num_creatures,
    int num_species)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    int sp = species_ids[idx];
    if (sp < 0 || sp >= num_species) return;

    atomicAdd(&species_counts[sp], 1);

    int genome_base = idx * GENOME_LENGTH;
    int centroid_base = sp * CENTROID_GENES;

    for (int g = 0; g < CENTROID_GENES; g++) {
        atomicAdd(&centroid_accum[centroid_base + g], genomes[genome_base + g]);
    }
}

__global__ void centroidNormalizeKernel(
    float* __restrict__ centroids,
    const float* __restrict__ centroid_accum,
    const int* __restrict__ species_counts,
    int num_species,
    float update_rate)
{
    int sp = blockIdx.x * blockDim.x + threadIdx.x;
    if (sp >= num_species) return;

    int count = species_counts[sp];
    if (count == 0) return;

    int base = sp * CENTROID_GENES;
    float inv_count = 1.0f / (float)count;

    for (int g = 0; g < CENTROID_GENES; g++) {
        float new_centroid = centroid_accum[base + g] * inv_count;
        float old_centroid = centroids[base + g];
        centroids[base + g] = old_centroid * (1.0f - update_rate) + new_centroid * update_rate;
    }
}

__global__ void populationCountKernel(
    const int* __restrict__ species_ids,
    int* __restrict__ species_populations,
    int num_creatures,
    int num_species)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    int sp = species_ids[idx];
    if (sp >= 0 && sp < num_species) {
        atomicAdd(&species_populations[sp], 1);
    }
}

void launchSpeciationCheckKernel(
    const float* genomes, int* species_ids,
    const float* species_centroids, const int* parent_species,
    int num_children, const int* child_indices,
    int* d_next_species_id, int* d_new_species_flags,
    const SpeciationParams& params, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_children + block - 1) / block;
    speciationCheckKernel<<<grid, block, 0, stream>>>(
        genomes, species_ids, species_centroids, parent_species,
        num_children, child_indices, d_next_species_id,
        d_new_species_flags,
        params.distance_threshold, params.comparison_genes);
}

void launchCentroidAccumulateKernel(
    const float* genomes, const int* species_ids,
    float* centroid_accum, int* species_counts,
    int num_creatures, int num_species, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    CUDA_CHECK(cudaMemsetAsync(centroid_accum, 0,
        num_species * CENTROID_GENES * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(species_counts, 0,
        num_species * sizeof(int), stream));
    centroidAccumulateKernel<<<grid, block, 0, stream>>>(
        genomes, species_ids, centroid_accum, species_counts,
        num_creatures, num_species);
}

void launchCentroidNormalizeKernel(
    float* centroids, const float* centroid_accum,
    const int* species_counts, int num_species,
    float update_rate, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_species + block - 1) / block;
    centroidNormalizeKernel<<<grid, block, 0, stream>>>(
        centroids, centroid_accum, species_counts,
        num_species, update_rate);
}

void launchPopulationCountKernel(
    const int* species_ids, int* species_populations,
    int num_creatures, int num_species, cudaStream_t stream)
{
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    CUDA_CHECK(cudaMemsetAsync(species_populations, 0,
        num_species * sizeof(int), stream));
    populationCountKernel<<<grid, block, 0, stream>>>(
        species_ids, species_populations, num_creatures, num_species);
}

}
}