#ifndef CREATURE_MANAGER_CUH
#define CREATURE_MANAGER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "creature_common.cuh"
#include "creature_spawner.cuh"
#include "creature_movement.cuh"
#include "creature_interaction.cuh"
#include "creature_lifecycle.cuh"
#include "stream_compaction.cuh"
#include "creature_renderer.cuh"
#include "../neural/neural_manager.cuh"
#include "../spatial/spatial_hash.cuh"

class CreatureManager {
public:
    CreatureData creatures;
    InteractionBuffers interaction;
    CompactionBuffers compaction;
    CreatureRenderData render_data;
    NeuralManager* neural;

    InteractionParams interaction_params;
    LifecycleParams lifecycle_params;

    curandState* d_rng_states;
    int* d_species_counter;
    float speciation_threshold;

    int current_alive;
    int max_creatures;

    cudaStream_t stream_lifecycle;
    cudaStream_t stream_interaction;
    cudaStream_t stream_movement;

    void init(int max_creatures, int initial_spawn, const float* d_heightmap, int world_size);
    void destroy();

    void update(
        float dt,
        const float* d_heightmap,
        float* d_vegetation,
        int world_size,
        const int* d_cell_start,
        const int* d_cell_end,
        const int* d_sorted_indices,
        int grid_size,
        float cell_size
    );

    void compact();
    void buildRenderData(const float* d_heightmap, int world_size);
    int getAliveCount();
};

#endif