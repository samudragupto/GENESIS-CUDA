#include "core/constants.cuh"
#include "core/cuda_utils.cuh"
#include "core/gpu_info.cuh"
#include "core/gpu_memory_pool.cuh"
#include "core/stream_manager.cuh"
#include "core/kernel_profiler.cuh"
#include "core/cuda_graphs.cuh"
#include "core/memory_optimizer.cuh"

#include "spatial/spatial_hash.cuh"
#include "spatial/sort_particles.cuh"
#include "spatial/neighbor_search.cuh"

#include "terrain/perlin_noise.cuh"
#include "terrain/terrain_generator.cuh"
#include "terrain/hydraulic_erosion.cuh"
#include "terrain/thermal_erosion.cuh"
#include "terrain/biome_classifier.cuh"
#include "terrain/terrain_renderer.cuh"

#include "fluid/fluid_manager.cuh"

#include "climate/climate_manager.cuh"

#include "genetics/genetic_manager.cuh"
#include "genetics/phylogenetic_tree.cuh"

#include "neural/neural_manager.cuh"

#include "creatures/creature_manager.cuh"

#include "ecosystem/ecosystem_manager.cuh"

#include "analytics/analytics_manager.cuh"

#include "rendering/render_manager.cuh"

#include "checkpoint/checkpoint_system.cuh"
#include "scenario/scenario_loader.h"
#include "benchmark/benchmark_suite.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern void printGPUInfo();
extern void launchPerlinHeightmap(float* d_heightmap, int world_size, int octaves, float lacunarity, float persistence, unsigned int seed, cudaStream_t stream);
extern void launchHydraulicErosion(float* d_heightmap, int world_size, int num_droplets, unsigned int seed, int max_lifetime, cudaStream_t stream);
extern void launchThermalErosion(float* d_heightmap, int world_size, float talus_angle, int iterations, cudaStream_t stream);



struct GenesisState {
    ScenarioConfig config;
    int current_tick;
    float simulation_time;
    float dt;
    int running;

    float* d_heightmap;
    float* d_normalmap;
    int* d_biome_map;

    SpatialHashGrid spatial_grid;
    ClimateManager climate;
    CreatureManager creature_mgr;
    EcosystemManager ecosystem;
    AnalyticsManager analytics;
    RenderManager renderer;
    CheckpointSystem checkpoint;
    KernelProfiler profiler;
    PhyloTreeGPU phylo_tree;

    cudaStream_t main_stream;
};


extern void initGenomeConstants();
void initializeGenesis(GenesisState& state, int argc, char** argv) {
    ScenarioLoader loader;
    loader.loadDefaults();
    loader.applyCommandLineArgs(argc, argv);
    state.config = loader.getConfig();
    loader.printConfig();

    printGPUInfo();

    state.current_tick = 0;
    state.simulation_time = 0.0f;
    state.dt = 1.0f;
    state.running = 1;

    CUDA_CHECK(cudaStreamCreate(&state.main_stream));

    state.profiler.init();

    int ws = state.config.world_size;
    size_t grid_bytes = (size_t)ws * ws * sizeof(float);

    CUDA_CHECK(cudaMalloc(&state.d_heightmap, grid_bytes));
    CUDA_CHECK(cudaMalloc(&state.d_normalmap, grid_bytes * 3));
    CUDA_CHECK(cudaMalloc(&state.d_biome_map, (size_t)ws * ws * sizeof(int)));

    printf("[GENESIS] Generating terrain...\n");

    launchPerlinHeightmap(
        state.d_heightmap, ws,
        state.config.terrain_octaves,
        state.config.terrain_lacunarity,
        state.config.terrain_persistence,
        state.config.terrain_seed,
        state.main_stream
    );
    CUDA_CHECK(cudaStreamSynchronize(state.main_stream));

    printf("[GENESIS] Applying erosion...\n");

    launchHydraulicErosion(
        state.d_heightmap, ws, 200000,
        state.config.terrain_seed,
        100, state.main_stream
    );
    CUDA_CHECK(cudaStreamSynchronize(state.main_stream));

    launchThermalErosion(
        state.d_heightmap, ws,
        0.01f, 50, state.main_stream
    );
    CUDA_CHECK(cudaStreamSynchronize(state.main_stream));

    printf("[GENESIS] Initializing climate...\n");

    state.climate.init(ws);
    state.climate.initFromTerrain(
        state.d_heightmap, ws,
        state.config.initial_temperature,
        state.config.temperature_variation
    );

    printf("[GENESIS] Initializing ecosystem...\n");

    int max_species = 10000;
    state.ecosystem.init(ws, max_species, state.config.max_creatures);
    state.ecosystem.initVegetation(
        state.d_heightmap,
        state.climate.getTemperaturePtr(),
        state.climate.getMoisturePtr()
    );

    printf("[GENESIS] Initializing spatial grid...\n");

    float cell_size = 4.0f;
    int grid_size = (int)((float)ws / cell_size) + 1;
    initSpatialHashGrid(state.spatial_grid, state.config.max_creatures, grid_size);

    printf("[GENESIS] Initializing creatures...\n");

    state.creature_mgr.init(
        state.config.max_creatures,
        state.config.initial_creatures,
        state.d_heightmap, ws
    );

    printf("[GENESIS] Initializing phylogenetic tree...\n");

    initPhyloTree(state.phylo_tree, max_species);

    printf("[GENESIS] Initializing analytics...\n");

    state.analytics.init(
        max_species, state.config.max_creatures, 10000,
        state.config.render_width / 4, state.config.render_height / 4,
        max_species
    );

    printf("[GENESIS] Initializing renderer...\n");

    state.renderer.init(
        state.config.render_width,
        state.config.render_height,
        ws
    );

    printf("[GENESIS] Initializing checkpoint system...\n");

    state.checkpoint.init(
        64 * 1024 * 1024,
        state.config.output_directory,
        state.config.auto_save_interval
    );

    printf("[GENESIS] Initialization complete!\n\n");
}

void simulationTick(GenesisState& state) {
    int prof_climate = state.profiler.registerKernel("Climate");
    int prof_spatial = state.profiler.registerKernel("SpatialHash");
    int prof_creatures = state.profiler.registerKernel("Creatures");
    int prof_ecosystem = state.profiler.registerKernel("Ecosystem");
    int prof_analytics = state.profiler.registerKernel("Analytics");

    state.profiler.beginProfile(prof_climate, state.main_stream);
    state.climate.update(
        state.dt, state.d_heightmap,
        state.config.world_size,
        state.current_tick
    );
    state.profiler.endProfile(prof_climate, state.main_stream);

    state.profiler.beginProfile(prof_spatial, state.main_stream);

    int alive = state.creature_mgr.getAliveCount();

    if (alive > 0) {
        updateSpatialHashGrid(
            state.spatial_grid,
            state.creature_mgr.creatures.d_pos_x,
            state.creature_mgr.creatures.d_pos_y,
            state.creature_mgr.creatures.d_alive,
            alive,
            (float)state.config.world_size,
            state.main_stream
        );
    }
    state.profiler.endProfile(prof_spatial, state.main_stream);

    state.profiler.beginProfile(prof_creatures, state.main_stream);
    state.creature_mgr.update(
        state.dt,
        state.d_heightmap,
        state.ecosystem.getVegetationPtr(),
        state.config.world_size,
        state.spatial_grid.d_cell_start,
        state.spatial_grid.d_cell_end,
        state.spatial_grid.d_sorted_indices,
        state.spatial_grid.grid_size,
        state.spatial_grid.cell_size
    );
    state.profiler.endProfile(prof_creatures, state.main_stream);

    state.profiler.beginProfile(prof_ecosystem, state.main_stream);
    state.ecosystem.update(
        state.dt,
        state.creature_mgr.creatures,
        state.creature_mgr.getAliveCount(),
        state.d_heightmap,
        state.climate.getTemperaturePtr(),
        state.climate.getMoisturePtr(),
        state.spatial_grid.d_cell_start,
        state.spatial_grid.d_cell_end,
        state.spatial_grid.d_sorted_indices,
        state.spatial_grid.grid_size,
        state.spatial_grid.cell_size,
        state.creature_mgr.d_rng_states
    );
    state.profiler.endProfile(prof_ecosystem, state.main_stream);

    if (state.config.enable_disease &&
        state.current_tick == state.config.disease_start_tick) {
        state.ecosystem.triggerDisease(
            state.creature_mgr.getAliveCount(),
            state.creature_mgr.d_rng_states
        );
        printf("[GENESIS] Disease outbreak triggered at tick %d!\n", state.current_tick);
    }

    if (state.current_tick % 50 == 0) {
        state.creature_mgr.compact();
    }

    state.profiler.beginProfile(prof_analytics, state.main_stream);
    state.analytics.update(
        state.creature_mgr.creatures,
        state.creature_mgr.getAliveCount(),
        state.current_tick,
        state.ecosystem.pop_stats
    );
    state.profiler.endProfile(prof_analytics, state.main_stream);

    CUDA_CHECK(cudaStreamSynchronize(state.main_stream));
    state.profiler.collectResults();

    state.current_tick++;
    state.simulation_time += state.dt;
}

void renderFrame(GenesisState& state) {
    state.renderer.beginFrame();
    state.renderer.updateDayNight(state.dt);
    state.renderer.renderSky();
    state.renderer.renderClouds(state.simulation_time);
    state.renderer.renderTerrain(state.d_heightmap,
        state.ecosystem.getVegetationPtr(), state.config.world_size);
    state.creature_mgr.buildRenderData(state.d_heightmap, state.config.world_size);
    state.renderer.applyDayNightLighting();
    state.renderer.applyPostProcessing();

    AnalyticsSnapshot snap = state.analytics.getSnapshot();
    state.renderer.renderUI(snap,
        state.profiler.getTotalFrameTime(),
        state.profiler.getKernelTime("Creatures"),
        state.profiler.getKernelTime("Climate"));

    state.renderer.endFrame();
}

void handleCheckpoint(GenesisState& state) {
    if (state.checkpoint.shouldAutoSave(state.current_tick)) {
        state.checkpoint.autoSave(
            state.current_tick, state.simulation_time,
            state.creature_mgr.creatures,
            state.creature_mgr.getAliveCount(),
            state.config.world_size,
            state.ecosystem.eco_grid,
            state.climate.getTemperaturePtr(),
            state.climate.getMoisturePtr(),
            nullptr, nullptr,
            state.phylo_tree
        );
    }
}

void printStatusReport(GenesisState& state) {
    AnalyticsSnapshot snap = state.analytics.getSnapshot();
    printf("[Tick %6d] Alive: %6d | Species: %4d | AvgE: %.3f | Shannon: %.3f | Frame: %.2fms\n",
           state.current_tick,
           snap.total_alive,
           snap.total_species,
           snap.avg_energy,
           snap.shannon_diversity,
           state.profiler.getTotalFrameTime());
}

void shutdownGenesis(GenesisState& state) {
    printf("\n[GENESIS] Shutting down...\n");

    printf("[GENESIS] Exporting final data...\n");
    state.analytics.exportData(
        state.creature_mgr.creatures,
        state.creature_mgr.getAliveCount(),
        state.config.output_directory
    );

    state.profiler.printReport();
    state.profiler.printTopN(10);

    state.creature_mgr.destroy();
    state.ecosystem.destroy();
    state.climate.destroy();
    state.analytics.destroy();
    state.renderer.destroy();
    state.checkpoint.destroy();
    state.profiler.destroy();
    freeSpatialHashGrid(state.spatial_grid);
    freePhyloTree(state.phylo_tree);

    cudaFree(state.d_heightmap);
    cudaFree(state.d_normalmap);
    cudaFree(state.d_biome_map);

    cudaStreamDestroy(state.main_stream);

    printf("[GENESIS] Shutdown complete.\n");
}

int main(int argc, char** argv) {
    printf("==================================================\n");
    printf("  GENESIS: GPU-Accelerated Planetary Evolution\n");
    printf("  & Ecosystem Simulator\n");
    printf("==================================================\n\n");

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--benchmark") == 0) {
            BenchmarkSuite bench;
            bench.init(5, 50);
            bench.runAll(1024, 100000);
            bench.exportCSV("benchmark_results.csv");
            bench.destroy();
            return 0;
        }
        if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: genesis [options]\n\n");
            printf("Options:\n");
            printf("  --scenario <name>     Load preset scenario\n");
            printf("                        (island, continent, ocean, desert,\n");
            printf("                         iceage, pandemic, extinction, radiation)\n");
            printf("  --config <file>       Load configuration file\n");
            printf("  --world-size <N>      Set world size (NxN)\n");
            printf("  --creatures <N>       Set initial creature count\n");
            printf("  --max-creatures <N>   Set maximum creature count\n");
            printf("  --seed <N>            Set terrain random seed\n");
            printf("  --output <dir>        Set output directory\n");
            printf("  --benchmark           Run benchmark suite\n");
            printf("  --help                Show this help\n");
            return 0;
        }
    }

    GenesisState state;
    initializeGenesis(state, argc, argv);

    int max_ticks = 100000;
    int report_interval = 100;
    int render_interval = 1;

    printf("[GENESIS] Starting simulation loop...\n\n");

    cudaEvent_t frame_start, frame_stop;
    CUDA_CHECK(cudaEventCreate(&frame_start));
    CUDA_CHECK(cudaEventCreate(&frame_stop));

    while (state.running && state.current_tick < max_ticks) {
        CUDA_CHECK(cudaEventRecord(frame_start));

        simulationTick(state);

        if (state.current_tick % render_interval == 0) {
            renderFrame(state);
        }

        handleCheckpoint(state);

        CUDA_CHECK(cudaEventRecord(frame_stop));
        CUDA_CHECK(cudaEventSynchronize(frame_stop));

        if (state.current_tick % report_interval == 0) {
            printStatusReport(state);
        }

        int alive = state.creature_mgr.getAliveCount();
        if (alive <= 0) {
            printf("[GENESIS] All creatures have died. Simulation ended at tick %d.\n",
                   state.current_tick);
            break;
        }

        if (alive >= state.config.max_creatures * 95 / 100) {
            printf("[GENESIS] Population near capacity (%d). Compacting...\n", alive);
            state.creature_mgr.compact();
        }
    }

    printf("\n[GENESIS] Simulation completed after %d ticks.\n", state.current_tick);

    cudaEventDestroy(frame_start);
    cudaEventDestroy(frame_stop);

    shutdownGenesis(state);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}