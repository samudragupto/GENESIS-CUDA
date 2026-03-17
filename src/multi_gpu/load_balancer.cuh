#ifndef LOAD_BALANCER_CUH
#define LOAD_BALANCER_CUH

#include <cuda_runtime.h>
#include "multi_gpu_manager.cuh"

struct LoadMetrics {
    int creature_counts[MAX_GPUS];
    float compute_times[MAX_GPUS];
    float transfer_times[MAX_GPUS];
    float load_imbalance;
    int num_gpus;
};

struct BalancerConfig {
    float imbalance_threshold;
    int min_region_width;
    int rebalance_interval;
    int tick_counter;
};

class LoadBalancer {
public:
    LoadMetrics metrics;
    BalancerConfig config;
    int boundary_positions[MAX_GPUS + 1];

    void init(int num_gpus, int world_size);

    void collectMetrics(const MultiGPUManager& mgr);

    bool shouldRebalance();

    void computeNewPartition(int world_size);

    void applyPartition(MultiGPUManager& mgr);

    void tick();

    void printMetrics() const;
};

void launchCountCreaturesPerRegion(
    const float* d_pos_x,
    const int* d_alive,
    int* d_region_counts,
    const int* d_boundaries,
    int num_regions,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif