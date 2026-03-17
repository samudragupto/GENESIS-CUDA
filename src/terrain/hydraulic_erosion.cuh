#pragma once
#include <cuda_runtime.h>
#include "terrain/terrain_generator.cuh"

namespace genesis {

struct ErosionParams {
    int     maxDropletLifetime;
    float   inertia;
    float   sedimentCapacity;
    float   depositSpeed;
    float   erodeSpeed;
    float   evaporateSpeed;
    float   gravity;
    float   minSlope;
    int     erosionRadius;
};

class HydraulicErosion {
public:
    HydraulicErosion();
    ~HydraulicErosion();

    void    initialize(int mapWidth, int mapHeight);
    void    destroy();

    // Erode terrain with N droplets
    void    erode(float* d_heightmap, int numDroplets,
                  unsigned int seed, cudaStream_t stream = 0);

    // Set custom erosion parameters
    void    setParams(const ErosionParams& params);
    const ErosionParams& getParams() const { return params_; }

private:
    ErosionParams   params_;
    int             mapWidth_;
    int             mapHeight_;
    bool            initialized_;

    // Precomputed erosion brush weights
    float*          d_brushWeights_;
    int*            d_brushOffsets_;
    int             brushLength_;

    void    precomputeBrush();
};

} // namespace genesis