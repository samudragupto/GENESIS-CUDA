#include "multi_gpu_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <cstdio>
#include <cstring>

static void allocateMigrationBuffer(MigrationBuffer& buf, int capacity, int device) {
    buf.capacity = capacity;
    cudaSetDevice(device);
    CUDA_CHECK(cudaMalloc(&buf.d_pos_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_pos_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_vel_x, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_vel_y, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_energy, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_health, capacity * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_age, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_species_id, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_state, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_genomes, (size_t)capacity * GENOME_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_neural_weights, (size_t)capacity * NEURAL_WEIGHT_COUNT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.d_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(buf.d_count, 0, sizeof(int)));
}

static void freeMigrationBuffer(MigrationBuffer& buf) {
    cudaFree(buf.d_pos_x);
    cudaFree(buf.d_pos_y);
    cudaFree(buf.d_vel_x);
    cudaFree(buf.d_vel_y);
    cudaFree(buf.d_energy);
    cudaFree(buf.d_health);
    cudaFree(buf.d_age);
    cudaFree(buf.d_species_id);
    cudaFree(buf.d_state);
    cudaFree(buf.d_genomes);
    cudaFree(buf.d_neural_weights);
    cudaFree(buf.d_count);
}

void MultiGPUManager::detectTopology() {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    topology.num_devices = device_count;

    for (int i = 0; i < device_count; i++) {
        for (int j = 0; j < device_count; j++) {
            if (i == j) {
                topology.can_access[i][j] = 1;
                continue;
            }
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
            topology.can_access[i][j] = can_access;
        }
    }
}

void MultiGPUManager::enableP2P() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j && topology.can_access[i][j]) {
                cudaDeviceEnablePeerAccess(j, 0);
            }
        }
    }
}

void MultiGPUManager::disableP2P() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < num_gpus; j++) {
            if (i != j && topology.can_access[i][j]) {
                cudaDeviceDisablePeerAccess(j);
            }
        }
    }
}

void MultiGPUManager::init(int num_devices, int ws, int max_creatures_per_gpu,
                            int hw) {
    detectTopology();

    num_gpus = (num_devices <= topology.num_devices) ? num_devices : topology.num_devices;
    world_size = ws;
    halo_width = hw;
    total_creatures = 0;

    enableP2P();

    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        regions[i].device_id = i;
        regions[i].world_size = ws;
        regions[i].halo_width = hw;
        regions[i].creature_count = 0;
        regions[i].creature_capacity = max_creatures_per_gpu;

        CUDA_CHECK(cudaStreamCreate(&regions[i].compute_stream));
        CUDA_CHECK(cudaStreamCreate(&regions[i].transfer_stream));
        CUDA_CHECK(cudaEventCreate(&regions[i].compute_done));
        CUDA_CHECK(cudaEventCreate(&regions[i].transfer_done));

        int region_w = ws / num_gpus;
        int halo_cells = hw * ws;

        CUDA_CHECK(cudaMalloc(&regions[i].d_heightmap, (size_t)ws * ws * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_vegetation, (size_t)ws * ws * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_temperature, (size_t)ws * ws * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_moisture, (size_t)ws * ws * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_send_left, halo_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_send_right, halo_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_send_top, region_w * hw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_send_bottom, region_w * hw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_recv_left, halo_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_recv_right, halo_cells * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_recv_top, region_w * hw * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&regions[i].d_halo_recv_bottom, region_w * hw * sizeof(float)));

        allocateCreatureData(regions[i].creatures, max_creatures_per_gpu);

        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                allocateMigrationBuffer(migration_out[i][j],
                    max_creatures_per_gpu / 100 + 256, i);
            }
        }
    }

    cudaSetDevice(0);
}

void MultiGPUManager::destroy() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        cudaStreamDestroy(regions[i].compute_stream);
        cudaStreamDestroy(regions[i].transfer_stream);
        cudaEventDestroy(regions[i].compute_done);
        cudaEventDestroy(regions[i].transfer_done);

        cudaFree(regions[i].d_heightmap);
        cudaFree(regions[i].d_vegetation);
        cudaFree(regions[i].d_temperature);
        cudaFree(regions[i].d_moisture);
        cudaFree(regions[i].d_halo_send_left);
        cudaFree(regions[i].d_halo_send_right);
        cudaFree(regions[i].d_halo_send_top);
        cudaFree(regions[i].d_halo_send_bottom);
        cudaFree(regions[i].d_halo_recv_left);
        cudaFree(regions[i].d_halo_recv_right);
        cudaFree(regions[i].d_halo_recv_top);
        cudaFree(regions[i].d_halo_recv_bottom);

        freeCreatureData(regions[i].creatures);

        for (int j = 0; j < num_gpus; j++) {
            if (i != j) {
                freeMigrationBuffer(migration_out[i][j]);
            }
        }
    }

    disableP2P();
    cudaSetDevice(0);
}

void MultiGPUManager::partitionWorld(int split_mode) {
    int region_w = world_size / num_gpus;

    for (int i = 0; i < num_gpus; i++) {
        if (split_mode == 0) {
            regions[i].region_start_x = i * region_w;
            regions[i].region_end_x = (i + 1) * region_w;
            regions[i].region_start_y = 0;
            regions[i].region_end_y = world_size;
        } else {
            int rows = (int)sqrtf((float)num_gpus);
            int cols = (num_gpus + rows - 1) / rows;
            int row = i / cols;
            int col = i % cols;
            int rh = world_size / rows;
            int rw = world_size / cols;
            regions[i].region_start_x = col * rw;
            regions[i].region_end_x = (col + 1) * rw;
            regions[i].region_start_y = row * rh;
            regions[i].region_end_y = (row + 1) * rh;
        }

        if (i == num_gpus - 1) {
            regions[i].region_end_x = world_size;
        }
    }
}

void MultiGPUManager::distributeHeightmap(const float* h_heightmap) {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        CUDA_CHECK(cudaMemcpy(regions[i].d_heightmap, h_heightmap,
            (size_t)world_size * world_size * sizeof(float),
            cudaMemcpyHostToDevice));
    }
    cudaSetDevice(0);
}

void MultiGPUManager::distributeCreatures(const float* h_pos_x, const float* h_pos_y,
                                           int total_count) {
    int* assignments = new int[total_count];

    for (int c = 0; c < total_count; c++) {
        float px = h_pos_x[c];
        assignments[c] = 0;
        for (int g = 0; g < num_gpus; g++) {
            if (px >= (float)regions[g].region_start_x &&
                px < (float)regions[g].region_end_x) {
                assignments[c] = g;
                break;
            }
        }
    }

    int counts[MAX_GPUS];
    memset(counts, 0, sizeof(counts));
    for (int c = 0; c < total_count; c++) {
        counts[assignments[c]]++;
    }

    for (int g = 0; g < num_gpus; g++) {
        regions[g].creature_count = counts[g];
    }

    total_creatures = total_count;
    delete[] assignments;
}

__global__ void packHaloKernel(
    const float* __restrict__ field,
    float* __restrict__ halo_buf,
    int start_col,
    int num_cols,
    int world_size
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= world_size) return;

    for (int c = 0; c < num_cols; c++) {
        halo_buf[y * num_cols + c] = field[y * world_size + start_col + c];
    }
}

__global__ void unpackHaloKernel(
    float* __restrict__ field,
    const float* __restrict__ halo_buf,
    int start_col,
    int num_cols,
    int world_size
) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= world_size) return;

    for (int c = 0; c < num_cols; c++) {
        field[y * world_size + start_col + c] = halo_buf[y * num_cols + c];
    }
}

void MultiGPUManager::exchangeHalos() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        int block = 256;
        int grid = (world_size + block - 1) / block;

        int right_col = regions[i].region_end_x - halo_width;
        packHaloKernel<<<grid, block, 0, regions[i].transfer_stream>>>(
            regions[i].d_vegetation, regions[i].d_halo_send_right,
            right_col, halo_width, world_size
        );

        int left_col = regions[i].region_start_x;
        packHaloKernel<<<grid, block, 0, regions[i].transfer_stream>>>(
            regions[i].d_vegetation, regions[i].d_halo_send_left,
            left_col, halo_width, world_size
        );

        CUDA_CHECK(cudaEventRecord(regions[i].transfer_done, regions[i].transfer_stream));
    }

    for (int i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaStreamSynchronize(regions[i].transfer_stream));
    }

    for (int i = 0; i < num_gpus - 1; i++) {
        int src = i;
        int dst = i + 1;
        int halo_size = world_size * halo_width * sizeof(float);

        if (topology.can_access[dst][src]) {
            cudaSetDevice(dst);
            CUDA_CHECK(cudaMemcpyPeerAsync(
                regions[dst].d_halo_recv_left, dst,
                regions[src].d_halo_send_right, src,
                halo_size, regions[dst].transfer_stream
            ));
        } else {
            float* h_temp;
            CUDA_CHECK(cudaMallocHost(&h_temp, halo_size));
            cudaSetDevice(src);
            CUDA_CHECK(cudaMemcpy(h_temp, regions[src].d_halo_send_right,
                halo_size, cudaMemcpyDeviceToHost));
            cudaSetDevice(dst);
            CUDA_CHECK(cudaMemcpy(regions[dst].d_halo_recv_left, h_temp,
                halo_size, cudaMemcpyHostToDevice));
            cudaFreeHost(h_temp);
        }

        if (topology.can_access[src][dst]) {
            cudaSetDevice(src);
            CUDA_CHECK(cudaMemcpyPeerAsync(
                regions[src].d_halo_recv_right, src,
                regions[dst].d_halo_send_left, dst,
                halo_size, regions[src].transfer_stream
            ));
        }
    }

    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        int block = 256;
        int grid = (world_size + block - 1) / block;

        if (i > 0) {
            int left_col = regions[i].region_start_x - halo_width;
            if (left_col >= 0) {
                unpackHaloKernel<<<grid, block, 0, regions[i].transfer_stream>>>(
                    regions[i].d_vegetation, regions[i].d_halo_recv_left,
                    left_col, halo_width, world_size
                );
            }
        }

        if (i < num_gpus - 1) {
            int right_col = regions[i].region_end_x;
            if (right_col + halo_width <= world_size) {
                unpackHaloKernel<<<grid, block, 0, regions[i].transfer_stream>>>(
                    regions[i].d_vegetation, regions[i].d_halo_recv_right,
                    right_col, halo_width, world_size
                );
            }
        }
    }

    synchronizeAll();
    cudaSetDevice(0);
}

__global__ void findMigrantsKernel(
    const float* __restrict__ pos_x,
    const int* __restrict__ alive,
    int* __restrict__ migrate_flag,
    int region_start_x,
    int region_end_x,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) { migrate_flag[idx] = 0; return; }

    float px = pos_x[idx];
    migrate_flag[idx] = (px < (float)region_start_x || px >= (float)region_end_x) ? 1 : 0;
}

void MultiGPUManager::migrateCreatures() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);

        int* d_migrate_flag;
        CUDA_CHECK(cudaMalloc(&d_migrate_flag, regions[i].creature_count * sizeof(int)));

        int block = 256;
        int grid = (regions[i].creature_count + block - 1) / block;

        findMigrantsKernel<<<grid, block, 0, regions[i].compute_stream>>>(
            regions[i].creatures.d_pos_x,
            regions[i].creatures.d_alive,
            d_migrate_flag,
            regions[i].region_start_x,
            regions[i].region_end_x,
            regions[i].creature_count
        );

        CUDA_CHECK(cudaStreamSynchronize(regions[i].compute_stream));
        cudaFree(d_migrate_flag);
    }

    synchronizeAll();
    cudaSetDevice(0);
}

void MultiGPUManager::synchronizeAll() {
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        CUDA_CHECK(cudaStreamSynchronize(regions[i].compute_stream));
        CUDA_CHECK(cudaStreamSynchronize(regions[i].transfer_stream));
    }
    cudaSetDevice(0);
}

void MultiGPUManager::gatherResults(float* h_heightmap_out, int* h_creature_counts) {
    cudaSetDevice(0);
    CUDA_CHECK(cudaMemcpy(h_heightmap_out, regions[0].d_heightmap,
        (size_t)world_size * world_size * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < num_gpus; i++) {
        h_creature_counts[i] = regions[i].creature_count;
    }
}

void MultiGPUManager::gatherCreaturePositions(float* h_all_pos_x, float* h_all_pos_y,
                                                int* h_total) {
    int offset = 0;
    for (int i = 0; i < num_gpus; i++) {
        cudaSetDevice(i);
        int count = regions[i].creature_count;
        if (count > 0) {
            CUDA_CHECK(cudaMemcpy(h_all_pos_x + offset, regions[i].creatures.d_pos_x,
                count * sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_all_pos_y + offset, regions[i].creatures.d_pos_y,
                count * sizeof(float), cudaMemcpyDeviceToHost));
        }
        offset += count;
    }
    *h_total = offset;
    cudaSetDevice(0);
}

int MultiGPUManager::getTotalCreatures() {
    int total = 0;
    for (int i = 0; i < num_gpus; i++) {
        total += regions[i].creature_count;
    }
    return total;
}

void MultiGPUManager::printStatus() const {
    printf("\n=== Multi-GPU Status ===\n");
    printf("GPUs: %d | World: %dx%d | Halo: %d\n",
           num_gpus, world_size, world_size, halo_width);
    for (int i = 0; i < num_gpus; i++) {
        printf("  GPU %d: Region [%d-%d, %d-%d] | Creatures: %d\n",
               i, regions[i].region_start_x, regions[i].region_end_x,
               regions[i].region_start_y, regions[i].region_end_y,
               regions[i].creature_count);
    }

    printf("  P2P Topology:\n");
    for (int i = 0; i < num_gpus; i++) {
        printf("    GPU %d: ", i);
        for (int j = 0; j < num_gpus; j++) {
            printf("%d ", topology.can_access[i][j]);
        }
        printf("\n");
    }
    printf("========================\n");
}