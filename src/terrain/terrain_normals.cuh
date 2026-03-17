#pragma once
#include <cuda_runtime.h>

namespace genesis {

class TerrainNormals {
public:
    // Compute normal vectors from heightmap using Sobel filter
    static void compute(const float* d_heightmap,
                        float* d_normalX, float* d_normalY, float* d_normalZ,
                        int width, int height,
                        float heightScale = 1.0f,
                        cudaStream_t stream = 0);
};

} // namespace genesis