#ifndef CHECKPOINT_SYSTEM_CUH
#define CHECKPOINT_SYSTEM_CUH

#include <cuda_runtime.h>
#include "../creatures/creature_common.cuh"
#include "../ecosystem/ecosystem_common.cuh"
#include "../analytics/analytics_common.cuh"
#include "../genetics/phylogenetic_tree.cuh"
#include "../climate/climate_common.cuh"
#include "../core/constants.cuh"

#define CHECKPOINT_MAGIC 0x47454E53
#define CHECKPOINT_VERSION 1

struct CheckpointHeader {
    unsigned int magic;
    int version;
    int world_size;
    int max_creatures;
    int num_alive;
    int num_species;
    int tick;
    float simulation_time;
    size_t creature_data_offset;
    size_t ecosystem_data_offset;
    size_t climate_data_offset;
    size_t phylo_data_offset;
    size_t analytics_data_offset;
    size_t total_size;
};

class CheckpointSystem {
public:
    float* h_pinned_buffer;
    size_t buffer_size;
    int auto_save_interval;
    int last_save_tick;
    char save_directory[256];

    void init(size_t max_buffer_size, const char* save_dir, int auto_interval);
    void destroy();

    bool saveCheckpoint(
        const char* filename,
        int tick,
        float sim_time,
        const CreatureData& creatures,
        int num_alive,
        int world_size,
        const EcosystemGridData& eco_grid,
        const float* d_temperature,
        const float* d_moisture,
        const float* d_wind_x,
        const float* d_wind_y,
        const PhyloTreeGPU& phylo_tree,
        cudaStream_t stream = 0
    );

    bool loadCheckpoint(
        const char* filename,
        int& tick,
        float& sim_time,
        CreatureData& creatures,
        int& num_alive,
        int& world_size,
        EcosystemGridData& eco_grid,
        float* d_temperature,
        float* d_moisture,
        float* d_wind_x,
        float* d_wind_y,
        PhyloTreeGPU& phylo_tree,
        cudaStream_t stream = 0
    );

    bool shouldAutoSave(int current_tick);
    void autoSave(
        int tick, float sim_time,
        const CreatureData& creatures,
        int num_alive, int world_size,
        const EcosystemGridData& eco_grid,
        const float* d_temperature,
        const float* d_moisture,
        const float* d_wind_x,
        const float* d_wind_y,
        const PhyloTreeGPU& phylo_tree
    );

    void listCheckpoints(const char* directory);
};

#endif