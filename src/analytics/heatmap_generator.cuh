#ifndef HEATMAP_GENERATOR_CUH
#define HEATMAP_GENERATOR_CUH

#include <cuda_runtime.h>

enum HeatmapType {
    HEATMAP_DENSITY = 0,
    HEATMAP_TEMPERATURE = 1,
    HEATMAP_VEGETATION = 2,
    HEATMAP_MOISTURE = 3,
    HEATMAP_ENERGY = 4,
    HEATMAP_SPECIES = 5,
    HEATMAP_ELEVATION = 6
};

struct HeatmapData {
    float4* d_color_output;
    float*  d_scalar_buffer;
    int     width;
    int     height;
};

void allocateHeatmapData(HeatmapData& hm, int width, int height);
void freeHeatmapData(HeatmapData& hm);

void launchScalarToHeatmap(
    HeatmapData& hm,
    const float* d_scalar_field,
    int field_width,
    int field_height,
    float min_val,
    float max_val,
    HeatmapType type,
    cudaStream_t stream = 0
);

void launchGaussianBlurHeatmap(
    HeatmapData& hm,
    int radius,
    float sigma,
    cudaStream_t stream = 0
);

void launchOverlayHeatmap(
    float4* d_framebuffer,
    const float4* d_heatmap,
    int width,
    int height,
    float alpha,
    cudaStream_t stream = 0
);

#endif