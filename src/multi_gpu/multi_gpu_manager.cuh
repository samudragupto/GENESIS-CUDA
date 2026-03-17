#ifndef MULTI_GPU_MANAGER_CUH
#define MULTI_GPU_MANAGER_CUH

#include <cuda_runtime.h>
#include "../creatures/creature_common.cuh"
#include "../ecosystem/ecosystem_common.cuh"

#define MAX_GPUS 8

struct GPURegion {
    int device_id;
    int region_start_x;
    int region_end_x;
    int region_start_y;
    int region_end_y;
    int world_size;
    int halo_width;

    float* d_heightmap;
    float* d_vegetation;
    float* d_temperature;
    float* d_moisture;

    float* d_halo_send_left;
    float* d_halo_send_right;
    float* d_halo_send_top;
    float* d_halo_send_bottom;
    float* d_halo_recv_left;
    float* d_halo_recv_right;
    float* d_halo_recv_top;
    float* d_halo_recv_bottom;

    CreatureData creatures;
    int creature_count;
    int creature_capacity;

    cudaStream_t compute_stream;
    cudaStream_t transfer_stream;
    cudaEvent_t compute_done;
    cudaEvent_t transfer_done;
};

struct MigrationBuffer {
    float* d_pos_x;
    float* d_pos_y;
    float* d_vel_x;
    float* d_vel_y;
    float* d_energy;
    float* d_health;
    int*   d_age;
    int*   d_species_id;
    int*   d_state;
    float* d_genomes;
    float* d_neural_weights;
    int*   d_count;
    int    capacity;
};

struct P2PTopology {
    int can_access[MAX_GPUS][MAX_GPUS];
    int num_devices;
};

class MultiGPUManager {
public:
    GPURegion regions[MAX_GPUS];
    MigrationBuffer migration_out[MAX_GPUS][MAX_GPUS];
    P2PTopology topology;
    int num_gpus;
    int world_size;
    int halo_width;
    int total_creatures;

    void init(int num_devices, int world_size, int max_creatures_per_gpu,
              int halo_width);
    void destroy();

    void detectTopology();
    void enableP2P();
    void disableP2P();

    void partitionWorld(int split_mode);
    void distributeHeightmap(const float* h_heightmap);
    void distributeCreatures(const float* h_pos_x, const float* h_pos_y,
                             int total_count);

    void exchangeHalos();
    void migrateCreatures();

    void synchronizeAll();

    void gatherResults(float* h_heightmap_out, int* h_creature_counts);
    void gatherCreaturePositions(float* h_all_pos_x, float* h_all_pos_y,
                                  int* h_total);

    int getTotalCreatures();
    void printStatus() const;
};

#endif