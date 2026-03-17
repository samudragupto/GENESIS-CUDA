#include "creature_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void CreatureManager::init(int max_c, int initial_spawn, const float* d_heightmap, int world_size) {
    max_creatures = max_c;
    current_alive = 0;

    allocateCreatureData(creatures, max_creatures);
    allocateInteractionBuffers(interaction, max_creatures);
    allocateCompactionBuffers(compaction, max_creatures);
    allocateCreatureRenderData(render_data, max_creatures / 2);

    CUDA_CHECK(cudaMalloc(&d_rng_states, max_creatures * sizeof(curandState)));
    CUDA_CHECK(cudaMalloc(&d_species_counter, sizeof(int)));

    int init_species = 10;
    CUDA_CHECK(cudaMemcpy(d_species_counter, &init_species, sizeof(int), cudaMemcpyHostToDevice));

    speciation_threshold = 0.3f;

    int block = 256;
    int grid = (max_creatures + block - 1) / block;

    extern void launchInitRNG(curandState* states, int count, unsigned long long seed, cudaStream_t stream);
    launchInitRNG(d_rng_states, max_creatures, 42ULL, 0);

    CUDA_CHECK(cudaStreamCreate(&stream_lifecycle));
    CUDA_CHECK(cudaStreamCreate(&stream_interaction));
    CUDA_CHECK(cudaStreamCreate(&stream_movement));

    interaction_params.eat_radius = 2.0f;
    interaction_params.attack_radius = 3.0f;
    interaction_params.mate_radius = 4.0f;
    interaction_params.eat_energy_gain = 0.5f;
    interaction_params.attack_damage = 0.3f;
    interaction_params.attack_energy_cost = 0.05f;
    interaction_params.mate_energy_cost = 0.2f;
    interaction_params.cannibalism_penalty = 0.5f;

    lifecycle_params.base_energy_drain = 0.005f;
    lifecycle_params.age_energy_factor = 0.5f;
    lifecycle_params.health_regen_rate = 0.01f;
    lifecycle_params.starvation_damage = 0.1f;
    lifecycle_params.dt = 1.0f;

    neural = new NeuralManager();
    neural->init(max_creatures);

    launchInitialSpawn(creatures, initial_spawn, d_heightmap, world_size, d_rng_states, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    current_alive = initial_spawn;
}

void CreatureManager::destroy() {
    freeCreatureData(creatures);
    freeInteractionBuffers(interaction);
    freeCompactionBuffers(compaction);
    freeCreatureRenderData(render_data);
    cudaFree(d_rng_states);
    cudaFree(d_species_counter);
    if (neural) {
        neural->destroy();
        delete neural;
        neural = nullptr;
    }
    cudaStreamDestroy(stream_lifecycle);
    cudaStreamDestroy(stream_interaction);
    cudaStreamDestroy(stream_movement);
}

void CreatureManager::update(
    float dt,
    const float* d_heightmap,
    float* d_vegetation,
    int world_size,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size
) {
    if (current_alive <= 0) return;

    lifecycle_params.dt = dt;

    neural->runForwardPass(
        creatures, d_heightmap, d_vegetation,
        world_size, current_alive, 0
    );

    CreatureActions& actions = neural->actions;

    launchMovementKernel(
        creatures, actions, d_heightmap,
        world_size, dt, current_alive, stream_movement
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_movement));

    launchFeedingKernel(
        creatures, actions, d_vegetation,
        world_size, interaction_params,
        current_alive, stream_interaction
    );

    launchCombatKernel(
        creatures, actions,
        d_cell_start, d_cell_end, d_sorted_indices,
        grid_size, cell_size, interaction_params,
        current_alive, d_rng_states, stream_interaction
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_interaction));

    launchMatingKernel(
        creatures, actions, interaction,
        d_cell_start, d_cell_end, d_sorted_indices,
        grid_size, cell_size, interaction_params,
        current_alive, d_rng_states, stream_interaction
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_interaction));

    launchSpawnOffspring(
        creatures, interaction,
        current_alive, max_creatures - current_alive,
        d_rng_states, d_species_counter,
        speciation_threshold, 0
    );

    int spawn_count = 0;
    CUDA_CHECK(cudaMemcpy(&spawn_count, interaction.d_spawn_counter,
        sizeof(int), cudaMemcpyDeviceToHost));
    current_alive += spawn_count;
    if (current_alive > max_creatures) current_alive = max_creatures;

    if (spawn_count > 0) {
        launchGenomeToWeightsBatch(
            creatures.d_genomes + (size_t)(current_alive - spawn_count) * GENOME_SIZE,
            creatures.d_neural_weights + (size_t)(current_alive - spawn_count) * NEURAL_WEIGHT_COUNT,
            spawn_count, GENOME_SIZE, NEURAL_WEIGHT_COUNT, 0
        );
    }

    launchLifecycleKernel(creatures, lifecycle_params, current_alive, stream_lifecycle);
    launchCooldownKernel(creatures, dt, current_alive, stream_lifecycle);
    launchDeathKernel(creatures, current_alive, stream_lifecycle);
    CUDA_CHECK(cudaStreamSynchronize(stream_lifecycle));
}

void CreatureManager::compact() {
    if (current_alive <= 0) return;
    current_alive = launchStreamCompaction(creatures, compaction, current_alive, 0);
}

void CreatureManager::buildRenderData(const float* d_heightmap, int world_size) {
    launchBuildInstanceData(creatures, render_data, d_heightmap, world_size, current_alive, 0);
}

int CreatureManager::getAliveCount() {
    return current_alive;
}