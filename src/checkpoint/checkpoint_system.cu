#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "checkpoint_system.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <cstdio>
#include <cstring>

void CheckpointSystem::init(size_t max_buffer_size, const char* save_dir, int auto_interval) {
    buffer_size = max_buffer_size;
    auto_save_interval = auto_interval;
    last_save_tick = 0;
    strncpy(save_directory, save_dir, 255);
    save_directory[255] = '\0';

    CUDA_CHECK(cudaMallocHost(&h_pinned_buffer, buffer_size));
}

void CheckpointSystem::destroy() {
    if (h_pinned_buffer) {
        cudaFreeHost(h_pinned_buffer);
        h_pinned_buffer = nullptr;
    }
}

bool CheckpointSystem::saveCheckpoint(
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
    cudaStream_t stream
) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return false;

    CheckpointHeader header;
    memset(&header, 0, sizeof(header));
    header.magic = CHECKPOINT_MAGIC;
    header.version = CHECKPOINT_VERSION;
    header.world_size = world_size;
    header.max_creatures = creatures.max_creatures;
    header.num_alive = num_alive;
    header.tick = tick;
    header.simulation_time = sim_time;

    fwrite(&header, sizeof(CheckpointHeader), 1, fp);

    size_t creature_floats = (size_t)num_alive;
    size_t genome_floats = (size_t)num_alive * GENOME_SIZE;
    size_t weight_floats = (size_t)num_alive * NEURAL_WEIGHT_COUNT;

    float* h_buf = h_pinned_buffer;

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_pos_x,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_pos_y,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_vel_x,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_vel_y,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_energy,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_health,
        creature_floats * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_buf, sizeof(float), creature_floats, fp);

    int* h_int_buf = (int*)h_buf;
    CUDA_CHECK(cudaMemcpyAsync(h_int_buf, creatures.d_age,
        num_alive * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_int_buf, sizeof(int), num_alive, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_int_buf, creatures.d_species_id,
        num_alive * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_int_buf, sizeof(int), num_alive, fp);

    CUDA_CHECK(cudaMemcpyAsync(h_int_buf, creatures.d_state,
        num_alive * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fwrite(h_int_buf, sizeof(int), num_alive, fp);

    size_t grid_cells = (size_t)world_size * world_size;
    size_t chunk_size = buffer_size / sizeof(float);

    for (size_t offset = 0; offset < genome_floats; offset += chunk_size) {
        size_t remaining = genome_floats - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_genomes + offset,
            copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fwrite(h_buf, sizeof(float), copy_count, fp);
    }

    for (size_t offset = 0; offset < weight_floats; offset += chunk_size) {
        size_t remaining = weight_floats - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(h_buf, creatures.d_neural_weights + offset,
            copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fwrite(h_buf, sizeof(float), copy_count, fp);
    }

    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(h_buf, eco_grid.d_vegetation + offset,
            copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fwrite(h_buf, sizeof(float), copy_count, fp);
    }

    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_temperature + offset,
            copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fwrite(h_buf, sizeof(float), copy_count, fp);
    }

    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        CUDA_CHECK(cudaMemcpyAsync(h_buf, d_moisture + offset,
            copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        fwrite(h_buf, sizeof(float), copy_count, fp);
    }

    fclose(fp);
    last_save_tick = tick;
    printf("Checkpoint saved: %s (tick %d, %d creatures)\n", filename, tick, num_alive);
    return true;
}

bool CheckpointSystem::loadCheckpoint(
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
    cudaStream_t stream
) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) return false;

    CheckpointHeader header;
    fread(&header, sizeof(CheckpointHeader), 1, fp);

    if (header.magic != CHECKPOINT_MAGIC) {
        fclose(fp);
        return false;
    }

    if (header.version != CHECKPOINT_VERSION) {
        fclose(fp);
        return false;
    }

    tick = header.tick;
    sim_time = header.simulation_time;
    num_alive = header.num_alive;
    world_size = header.world_size;

    float* h_buf = h_pinned_buffer;
    size_t creature_floats = (size_t)num_alive;
    size_t chunk_size = buffer_size / sizeof(float);

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_pos_x, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_pos_y, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_vel_x, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_vel_y, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_energy, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    fread(h_buf, sizeof(float), creature_floats, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_health, h_buf,
        creature_floats * sizeof(float), cudaMemcpyHostToDevice, stream));

    int* h_int_buf = (int*)h_buf;
    fread(h_int_buf, sizeof(int), num_alive, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_age, h_int_buf,
        num_alive * sizeof(int), cudaMemcpyHostToDevice, stream));

    fread(h_int_buf, sizeof(int), num_alive, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_species_id, h_int_buf,
        num_alive * sizeof(int), cudaMemcpyHostToDevice, stream));

    fread(h_int_buf, sizeof(int), num_alive, fp);
    CUDA_CHECK(cudaMemcpyAsync(creatures.d_state, h_int_buf,
        num_alive * sizeof(int), cudaMemcpyHostToDevice, stream));

    size_t genome_floats = (size_t)num_alive * GENOME_SIZE;
    for (size_t offset = 0; offset < genome_floats; offset += chunk_size) {
        size_t remaining = genome_floats - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        fread(h_buf, sizeof(float), copy_count, fp);
        CUDA_CHECK(cudaMemcpyAsync(creatures.d_genomes + offset, h_buf,
            copy_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    size_t weight_floats = (size_t)num_alive * NEURAL_WEIGHT_COUNT;
    for (size_t offset = 0; offset < weight_floats; offset += chunk_size) {
        size_t remaining = weight_floats - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        fread(h_buf, sizeof(float), copy_count, fp);
        CUDA_CHECK(cudaMemcpyAsync(creatures.d_neural_weights + offset, h_buf,
            copy_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    size_t grid_cells = (size_t)world_size * world_size;
    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        fread(h_buf, sizeof(float), copy_count, fp);
        CUDA_CHECK(cudaMemcpyAsync(eco_grid.d_vegetation + offset, h_buf,
            copy_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        fread(h_buf, sizeof(float), copy_count, fp);
        CUDA_CHECK(cudaMemcpyAsync(d_temperature + offset, h_buf,
            copy_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    for (size_t offset = 0; offset < grid_cells; offset += chunk_size) {
        size_t remaining = grid_cells - offset;
        size_t copy_count = (remaining < chunk_size) ? remaining : chunk_size;
        fread(h_buf, sizeof(float), copy_count, fp);
        CUDA_CHECK(cudaMemcpyAsync(d_moisture + offset, h_buf,
            copy_count * sizeof(float), cudaMemcpyHostToDevice, stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // alive flags set individually below
    int one = 1;
    for (int i = 0; i < num_alive; i++) {
        CUDA_CHECK(cudaMemcpy(creatures.d_alive + i, &one, sizeof(int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(creatures.d_num_alive, &num_alive, sizeof(int), cudaMemcpyHostToDevice));

    fclose(fp);
    printf("Checkpoint loaded: %s (tick %d, %d creatures)\n", filename, tick, num_alive);
    return true;
}

bool CheckpointSystem::shouldAutoSave(int current_tick) {
    return (auto_save_interval > 0) &&
           (current_tick - last_save_tick >= auto_save_interval);
}

void CheckpointSystem::autoSave(
    int tick, float sim_time,
    const CreatureData& creatures,
    int num_alive, int world_size,
    const EcosystemGridData& eco_grid,
    const float* d_temperature,
    const float* d_moisture,
    const float* d_wind_x,
    const float* d_wind_y,
    const PhyloTreeGPU& phylo_tree
) {
    char path[512];
    snprintf(path, sizeof(path), "%s/autosave_tick_%06d.gen", save_directory, tick);
    saveCheckpoint(path, tick, sim_time, creatures, num_alive, world_size,
                   eco_grid, d_temperature, d_moisture, d_wind_x, d_wind_y,
                   phylo_tree, 0);
}

void CheckpointSystem::listCheckpoints(const char* directory) {
    printf("Checkpoint directory: %s\n", directory);
    printf("(File listing not implemented - use filesystem API)\n");
}