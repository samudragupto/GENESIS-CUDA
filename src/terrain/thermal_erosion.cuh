#pragma once
#include <cuda_runtime.h>

namespace genesis {

class ThermalErosion {
public:
    // Perform thermal erosion iterations on heightmap
    // Transfers material from steep slopes to neighbors
    static void erode(float* d_heightmap,
                      int width, int height,
                      int iterations,
                      float talusAngle,
                      float transferRate = 0.5f,
                      cudaStream_t stream = 0);
};

} // namespace genesis