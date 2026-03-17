#ifndef TERRAIN_GENERATOR_CUH
#define TERRAIN_GENERATOR_CUH

#include <cuda_runtime.h>

namespace genesis {

class TerrainGenerator {
public:
    int width_;
    int height_;
    int octaves_;
    float lacunarity_;
    float persistence_;
    float* d_heightmap_;

    TerrainGenerator() : width_(0), height_(0), octaves_(8), 
                         lacunarity_(2.0f), persistence_(0.5f), 
                         d_heightmap_(nullptr) {}

    void initialize(int w, int h, float* d_map) {
        width_ = w;
        height_ = h;
        d_heightmap_ = d_map;
    }

    void setParams(int oct, float lac, float pers) {
        octaves_ = oct;
        lacunarity_ = lac;
        persistence_ = pers;
    }

    void generate(unsigned int seed, cudaStream_t stream = 0);
};

} // namespace genesis

#endif