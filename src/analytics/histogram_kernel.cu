#include "histogram_kernel.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void geneHistogramKernel(
    const float* __restrict__ genomes,
    const int* __restrict__ alive,
    int* __restrict__ gene_counts,
    int num_creatures,
    int num_genes,
    int num_bins
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    const float* genome = genomes + idx * GENOME_SIZE;

    for (int g = 0; g < num_genes && g < GENOME_SIZE; g++) {
        float val = fminf(fmaxf(genome[g], 0.0f), 1.0f);
        int bin = (int)(val * (float)(num_bins - 1));
        bin = min(max(bin, 0), num_bins - 1);
        atomicAdd(&gene_counts[g * num_bins + bin], 1);
    }
}

__global__ void normalizeHistogramKernel(
    const int* __restrict__ gene_counts,
    float* __restrict__ gene_histograms,
    int num_genes,
    int num_bins,
    int num_alive
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_genes * num_bins;
    if (idx >= total) return;

    float norm = (num_alive > 0) ? (float)gene_counts[idx] / (float)num_alive : 0.0f;
    gene_histograms[idx] = norm;
}

__global__ void spatialDensityKernel(
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const int* __restrict__ alive,
    float* __restrict__ density_grid,
    int grid_resolution,
    int world_size,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    float scale = (float)grid_resolution / (float)world_size;
    int gx = (int)(pos_x[idx] * scale);
    int gy = (int)(pos_y[idx] * scale);
    gx = min(max(gx, 0), grid_resolution - 1);
    gy = min(max(gy, 0), grid_resolution - 1);

    atomicAdd(&density_grid[gy * grid_resolution + gx], 1.0f);
}

__global__ void energyDistHistKernel(
    const float* __restrict__ energy,
    const int* __restrict__ alive,
    int* __restrict__ energy_hist,
    int num_bins,
    float max_energy,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    float e = fminf(fmaxf(energy[idx], 0.0f), max_energy);
    int bin = (int)(e / max_energy * (float)(num_bins - 1));
    bin = min(max(bin, 0), num_bins - 1);
    atomicAdd(&energy_hist[bin], 1);
}

void launchGeneHistogram(
    GeneFrequencyData& gene_freq,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(gene_freq.d_gene_counts, 0,
        gene_freq.num_genes * gene_freq.num_bins * sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    geneHistogramKernel<<<grid, block, 0, stream>>>(
        creatures.d_genomes, creatures.d_alive,
        gene_freq.d_gene_counts, num_creatures,
        gene_freq.num_genes, gene_freq.num_bins
    );
}

void launchNormalizeHistogram(
    GeneFrequencyData& gene_freq,
    int num_alive,
    cudaStream_t stream
) {
    int total = gene_freq.num_genes * gene_freq.num_bins;
    int block = 256;
    int grid = (total + block - 1) / block;
    normalizeHistogramKernel<<<grid, block, 0, stream>>>(
        gene_freq.d_gene_counts, gene_freq.d_gene_histograms,
        gene_freq.num_genes, gene_freq.num_bins, num_alive
    );
}

void launchSpatialDensityHistogram(
    const CreatureData& creatures,
    float* d_density_grid,
    int grid_resolution,
    int world_size,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_density_grid, 0,
        grid_resolution * grid_resolution * sizeof(float), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    spatialDensityKernel<<<grid, block, 0, stream>>>(
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_alive, d_density_grid,
        grid_resolution, world_size, num_creatures
    );
}

void launchEnergyDistributionHistogram(
    const CreatureData& creatures,
    int* d_energy_hist,
    int num_bins,
    float max_energy,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_energy_hist, 0, num_bins * sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    energyDistHistKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_alive,
        d_energy_hist, num_bins, max_energy, num_creatures
    );
}