#include "load_balancer.cuh"
#include "../core/cuda_utils.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

void LoadBalancer::init(int num_gpus, int world_size) {
    memset(&metrics, 0, sizeof(metrics));
    metrics.num_gpus = num_gpus;

    config.imbalance_threshold = 0.2f;
    config.min_region_width = 64;
    config.rebalance_interval = 100;
    config.tick_counter = 0;

    int region_w = world_size / num_gpus;
    for (int i = 0; i <= num_gpus; i++) {
        boundary_positions[i] = i * region_w;
    }
    boundary_positions[num_gpus] = world_size;
}

void LoadBalancer::collectMetrics(const MultiGPUManager& mgr) {
    int total = 0;
    for (int i = 0; i < metrics.num_gpus; i++) {
        metrics.creature_counts[i] = mgr.regions[i].creature_count;
        total += metrics.creature_counts[i];
    }

    if (total == 0 || metrics.num_gpus <= 1) {
        metrics.load_imbalance = 0.0f;
        return;
    }

    float avg = (float)total / (float)metrics.num_gpus;
    float max_dev = 0.0f;

    for (int i = 0; i < metrics.num_gpus; i++) {
        float dev = fabsf((float)metrics.creature_counts[i] - avg) / avg;
        if (dev > max_dev) max_dev = dev;
    }

    metrics.load_imbalance = max_dev;
}

bool LoadBalancer::shouldRebalance() {
    if (config.tick_counter % config.rebalance_interval != 0) return false;
    return metrics.load_imbalance > config.imbalance_threshold;
}

void LoadBalancer::computeNewPartition(int world_size) {
    int total = 0;
    for (int i = 0; i < metrics.num_gpus; i++) {
        total += metrics.creature_counts[i];
    }

    if (total == 0) return;

    float target_per_gpu = (float)total / (float)metrics.num_gpus;

    int cumulative = 0;
    int current_gpu = 0;
    int new_boundaries[MAX_GPUS + 1];
    new_boundaries[0] = 0;

    for (int i = 0; i < metrics.num_gpus; i++) {
        cumulative += metrics.creature_counts[i];

        if (current_gpu < metrics.num_gpus - 1) {
            float expected = target_per_gpu * (float)(current_gpu + 1);
            if ((float)cumulative >= expected) {
                int old_boundary = boundary_positions[current_gpu + 1];
                int shift = (int)((float)(cumulative - (int)expected) /
                             (float)total * (float)world_size);
                shift = (shift > 0) ? -(shift / 2) : ((-shift) / 2);

                int new_pos = old_boundary + shift;
                new_pos = (new_pos < config.min_region_width) ?
                          config.min_region_width : new_pos;
                new_pos = (new_pos > world_size - config.min_region_width) ?
                          world_size - config.min_region_width : new_pos;

                new_boundaries[current_gpu + 1] = new_pos;
                current_gpu++;
            }
        }
    }
    new_boundaries[metrics.num_gpus] = world_size;

    for (int i = 0; i < metrics.num_gpus; i++) {
        if (new_boundaries[i + 1] - new_boundaries[i] < config.min_region_width) {
            return;
        }
    }

    for (int i = 0; i <= metrics.num_gpus; i++) {
        int old = boundary_positions[i];
        int target = new_boundaries[i];
        boundary_positions[i] = old + (int)((float)(target - old) * 0.3f);
    }
    boundary_positions[0] = 0;
    boundary_positions[metrics.num_gpus] = world_size;
}

void LoadBalancer::applyPartition(MultiGPUManager& mgr) {
    for (int i = 0; i < metrics.num_gpus; i++) {
        mgr.regions[i].region_start_x = boundary_positions[i];
        mgr.regions[i].region_end_x = boundary_positions[i + 1];
    }
}

void LoadBalancer::tick() {
    config.tick_counter++;
}

void LoadBalancer::printMetrics() const {
    printf("\n=== Load Balancer Metrics ===\n");
    printf("Imbalance: %.2f%% (threshold: %.2f%%)\n",
           metrics.load_imbalance * 100.0f,
           config.imbalance_threshold * 100.0f);

    for (int i = 0; i < metrics.num_gpus; i++) {
        int width = boundary_positions[i + 1] - boundary_positions[i];
        printf("  GPU %d: %d creatures | region [%d - %d] width=%d\n",
               i, metrics.creature_counts[i],
               boundary_positions[i], boundary_positions[i + 1], width);
    }
    printf("============================\n");
}

__global__ void countCreaturesPerRegionKernel(
    const float* __restrict__ pos_x,
    const int* __restrict__ alive,
    int* __restrict__ region_counts,
    const int* __restrict__ boundaries,
    int num_regions,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    float px = pos_x[idx];

    for (int r = 0; r < num_regions; r++) {
        if (px >= (float)boundaries[r] && px < (float)boundaries[r + 1]) {
            atomicAdd(&region_counts[r], 1);
            return;
        }
    }
}

void launchCountCreaturesPerRegion(
    const float* d_pos_x,
    const int* d_alive,
    int* d_region_counts,
    const int* d_boundaries,
    int num_regions,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(d_region_counts, 0, num_regions * sizeof(int), stream));

    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    countCreaturesPerRegionKernel<<<grid, block, 0, stream>>>(
        d_pos_x, d_alive, d_region_counts, d_boundaries,
        num_regions, num_creatures
    );
}