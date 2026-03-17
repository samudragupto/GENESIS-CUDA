#pragma once
#include <cuda_runtime.h>
#include "spatial/spatial_hash.cuh"

namespace genesis {

// ═══════════════════════════════════════════════════════
//  Result structure for neighbor queries
// ═══════════════════════════════════════════════════════

constexpr int MAX_NEIGHBORS_PER_PARTICLE = 64;

struct NeighborList {
    int*    d_neighborIndices;   // [maxParticles * MAX_NEIGHBORS_PER_PARTICLE]
    int*    d_neighborCounts;    // [maxParticles]
    float*  d_neighborDists;     // [maxParticles * MAX_NEIGHBORS_PER_PARTICLE]
};

class NeighborSearch {
public:
    NeighborSearch();
    ~NeighborSearch();

    void    initialize(int maxParticles);
    void    destroy();

    // Perform neighbor search using built spatial hash grid
    void    search(const SpatialHashGrid& grid,
                   const float* d_posX, const float* d_posY,
                   int numParticles, float searchRadius,
                   cudaStream_t stream = 0);

    // Access results
    const NeighborList& getResults() const { return neighbors_; }

    // Count-only query (no distance computation — faster)
    void    countOnly(const SpatialHashGrid& grid,
                      const float* d_posX, const float* d_posY,
                      int numParticles, float searchRadius,
                      cudaStream_t stream = 0);

private:
    NeighborList    neighbors_;
    int             maxParticles_;
    bool            initialized_;
};

} // namespace genesis