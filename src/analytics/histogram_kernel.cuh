#ifndef HISTOGRAM_KERNEL_CUH
#define HISTOGRAM_KERNEL_CUH

#include <cuda_runtime.h>
#include "analytics_common.cuh"
#include "../creatures/creature_common.cuh"

void launchGeneHistogram(
    GeneFrequencyData& gene_freq,
    const CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchNormalizeHistogram(
    GeneFrequencyData& gene_freq,
    int num_alive,
    cudaStream_t stream = 0
);

void launchSpatialDensityHistogram(
    const CreatureData& creatures,
    float* d_density_grid,
    int grid_resolution,
    int world_size,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchEnergyDistributionHistogram(
    const CreatureData& creatures,
    int* d_energy_hist,
    int num_bins,
    float max_energy,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif