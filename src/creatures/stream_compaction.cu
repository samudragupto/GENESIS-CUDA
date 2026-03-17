#include "stream_compaction.cuh"
#include "../core/cuda_utils.cuh"
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

void allocateCompactionBuffers(CompactionBuffers& buf, int capacity) {
    buf.capacity = capacity;
    CUDA_CHECK(cudaMalloc(&buf.d_scan_input, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_scan_output, capacity * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_temp_indices, capacity * sizeof(int)));
}

void freeCompactionBuffers(CompactionBuffers& buf) {
    cudaFree(buf.d_scan_input);
    cudaFree(buf.d_scan_output);
    cudaFree(buf.d_temp_indices);
}

__global__ void gatherScatterKernel(
    const float* __restrict__ src_pos_x,
    const float* __restrict__ src_pos_y,
    const float* __restrict__ src_vel_x,
    const float* __restrict__ src_vel_y,
    const float* __restrict__ src_energy,
    const float* __restrict__ src_health,
    const int*   __restrict__ src_age,
    const int*   __restrict__ src_species_id,
    const int*   __restrict__ src_state,
    const float* __restrict__ src_repro_cooldown,
    const float* __restrict__ src_genomes,
    const float* __restrict__ src_neural_weights,
    const int*   __restrict__ src_alive,
    const int*   __restrict__ scatter_indices,
    float* __restrict__ dst_pos_x,
    float* __restrict__ dst_pos_y,
    float* __restrict__ dst_vel_x,
    float* __restrict__ dst_vel_y,
    float* __restrict__ dst_energy,
    float* __restrict__ dst_health,
    int*   __restrict__ dst_age,
    int*   __restrict__ dst_species_id,
    int*   __restrict__ dst_state,
    float* __restrict__ dst_repro_cooldown,
    float* __restrict__ dst_genomes,
    float* __restrict__ dst_neural_weights,
    int*   __restrict__ dst_alive,
    int new_count,
    int genome_size,
    int weight_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= new_count) return;

    int src = scatter_indices[idx];

    dst_pos_x[idx] = src_pos_x[src];
    dst_pos_y[idx] = src_pos_y[src];
    dst_vel_x[idx] = src_vel_x[src];
    dst_vel_y[idx] = src_vel_y[src];
    dst_energy[idx] = src_energy[src];
    dst_health[idx] = src_health[src];
    dst_age[idx] = src_age[src];
    dst_species_id[idx] = src_species_id[src];
    dst_state[idx] = src_state[src];
    dst_repro_cooldown[idx] = src_repro_cooldown[src];
    dst_alive[idx] = 1;

    for (int g = 0; g < genome_size; g++) {
        dst_genomes[idx * genome_size + g] = src_genomes[src * genome_size + g];
    }

    for (int w = 0; w < weight_count; w++) {
        dst_neural_weights[idx * weight_count + w] = src_neural_weights[src * weight_count + w];
    }
}

__global__ void buildScatterMap(
    const int* __restrict__ alive,
    const int* __restrict__ prefix_sum,
    int* __restrict__ scatter_indices,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    if (alive[idx]) {
        int dest = prefix_sum[idx];
        scatter_indices[dest] = idx;
    }
}

int launchStreamCompaction(
    CreatureData& creatures,
    CompactionBuffers& compaction,
    int current_count,
    cudaStream_t stream
) {
    if (current_count <= 0) return 0;

    thrust::device_ptr<int> alive_ptr(creatures.d_alive);
    thrust::device_ptr<int> scan_out_ptr(compaction.d_scan_output);

    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        alive_ptr, alive_ptr + current_count,
        scan_out_ptr
    );

    int last_alive, last_scan;
    CUDA_CHECK(cudaMemcpyAsync(&last_alive, creatures.d_alive + current_count - 1,
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&last_scan, compaction.d_scan_output + current_count - 1,
        sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int new_count = last_scan + last_alive;
    if (new_count <= 0) return 0;
    if (new_count == current_count) return current_count;

    int block = 256;
    int grid = (current_count + block - 1) / block;

    buildScatterMap<<<grid, block, 0, stream>>>(
        creatures.d_alive, compaction.d_scan_output,
        compaction.d_temp_indices, current_count
    );

    float* temp_pos_x; CUDA_CHECK(cudaMalloc(&temp_pos_x, new_count * sizeof(float)));
    float* temp_pos_y; CUDA_CHECK(cudaMalloc(&temp_pos_y, new_count * sizeof(float)));
    float* temp_vel_x; CUDA_CHECK(cudaMalloc(&temp_vel_x, new_count * sizeof(float)));
    float* temp_vel_y; CUDA_CHECK(cudaMalloc(&temp_vel_y, new_count * sizeof(float)));
    float* temp_energy; CUDA_CHECK(cudaMalloc(&temp_energy, new_count * sizeof(float)));
    float* temp_health; CUDA_CHECK(cudaMalloc(&temp_health, new_count * sizeof(float)));
    int* temp_age; CUDA_CHECK(cudaMalloc(&temp_age, new_count * sizeof(int)));
    int* temp_species; CUDA_CHECK(cudaMalloc(&temp_species, new_count * sizeof(int)));
    int* temp_state; CUDA_CHECK(cudaMalloc(&temp_state, new_count * sizeof(int)));
    float* temp_repro; CUDA_CHECK(cudaMalloc(&temp_repro, new_count * sizeof(float)));
    float* temp_genomes; CUDA_CHECK(cudaMalloc(&temp_genomes, (size_t)new_count * GENOME_SIZE * sizeof(float)));
    float* temp_weights; CUDA_CHECK(cudaMalloc(&temp_weights, (size_t)new_count * NEURAL_WEIGHT_COUNT * sizeof(float)));
    int* temp_alive; CUDA_CHECK(cudaMalloc(&temp_alive, new_count * sizeof(int)));

    grid = (new_count + block - 1) / block;

    gatherScatterKernel<<<grid, block, 0, stream>>>(
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_vel_x, creatures.d_vel_y,
        creatures.d_energy, creatures.d_health,
        creatures.d_age, creatures.d_species_id,
        creatures.d_state, creatures.d_repro_cooldown,
        creatures.d_genomes, creatures.d_neural_weights,
        creatures.d_alive, compaction.d_temp_indices,
        temp_pos_x, temp_pos_y,
        temp_vel_x, temp_vel_y,
        temp_energy, temp_health,
        temp_age, temp_species,
        temp_state, temp_repro,
        temp_genomes, temp_weights,
        temp_alive,
        new_count, GENOME_SIZE, NEURAL_WEIGHT_COUNT
    );

    CUDA_CHECK(cudaMemcpyAsync(creatures.d_pos_x, temp_pos_x, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_pos_y, temp_pos_y, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_vel_x, temp_vel_x, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_vel_y, temp_vel_y, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_energy, temp_energy, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_health, temp_health, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_age, temp_age, new_count * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_species_id, temp_species, new_count * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_state, temp_state, new_count * sizeof(int), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_repro_cooldown, temp_repro, new_count * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_genomes, temp_genomes, (size_t)new_count * GENOME_SIZE * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_neural_weights, temp_weights, (size_t)new_count * NEURAL_WEIGHT_COUNT * sizeof(float), cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_alive, temp_alive, new_count * sizeof(int), cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(temp_pos_x); cudaFree(temp_pos_y);
    cudaFree(temp_vel_x); cudaFree(temp_vel_y);
    cudaFree(temp_energy); cudaFree(temp_health);
    cudaFree(temp_age); cudaFree(temp_species);
    cudaFree(temp_state); cudaFree(temp_repro);
    cudaFree(temp_genomes); cudaFree(temp_weights);
    cudaFree(temp_alive);

    CUDA_CHECK(cudaMemcpyAsync(creatures.d_num_alive, &new_count,
        sizeof(int), cudaMemcpyHostToDevice, stream));

    return new_count;
}