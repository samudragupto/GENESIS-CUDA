#ifndef BIOME_CLASSIFIER_CUH
#define BIOME_CLASSIFIER_CUH

#include <cuda_runtime.h>

void launchBiomeClassification(
    int* d_biome_map,
    const float* d_heightmap,
    const float* d_moisture_map,
    int world_size,
    float sea_level,
    cudaStream_t stream = 0
);

#endif