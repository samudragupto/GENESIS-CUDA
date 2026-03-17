cmake_minimum_required(VERSION 3.18)
project(GENESIS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)

option(GENESIS_MULTI_GPU "Enable multi-GPU support" OFF)
option(GENESIS_OPENGL "Enable OpenGL rendering" OFF)

find_package(CUDAToolkit REQUIRED)

set(CORE_SOURCES
    src/core/gpu_memory_pool.cu
    src/core/stream_manager.cu
    src/core/gpu_info.cu
    src/core/config_manager.cpp
    src/core/kernel_profiler.cu
    src/core/cuda_graphs.cu
    src/core/memory_optimizer.cu
)
set(SPATIAL_SOURCES
    src/spatial/spatial_hash.cu
    src/spatial/sort_particles.cu
    src/spatial/neighbor_search.cu
)
set(TERRAIN_SOURCES
    src/terrain/perlin_noise.cu
    src/terrain/terrain_generator.cu
    src/terrain/hydraulic_erosion.cu
    src/terrain/thermal_erosion.cu
    src/terrain/biome_classifier.cu
    src/terrain/terrain_normals.cu
    src/terrain/terrain_renderer.cu
)
set(FLUID_SOURCES
    src/fluid/sph_density.cu
    src/fluid/sph_forces.cu
    src/fluid/sph_integration.cu
    src/fluid/sph_boundary.cu
    src/fluid/fluid_terrain_coupling.cu
    src/fluid/fluid_renderer.cu
    src/fluid/fluid_manager.cu
)
set(CLIMATE_SOURCES
    src/climate/solar_model.cu
    src/climate/heat_diffusion.cu
    src/climate/wind_simulation.cu
    src/climate/humidity_transport.cu
    src/climate/precipitation.cu
    src/climate/seasonal_cycle.cu
    src/climate/climate_manager.cu
)
set(GENETICS_SOURCES
    src/genetics/genome.cu
    src/genetics/curand_states.cu
    src/genetics/crossover_kernels.cu
    src/genetics/mutation_kernels.cu
    src/genetics/speciation.cu
    src/genetics/phylogenetic_tree.cu
    src/genetics/fitness_evaluation.cu
    src/genetics/genetic_manager.cu
)
set(NEURAL_SOURCES
    src/neural/sensory_input.cu
    src/neural/neural_forward.cu
    src/neural/decision_extractor.cu
    src/neural/genome_to_weights.cu
    src/neural/neural_manager.cu
)
set(CREATURE_SOURCES
    src/creatures/creature_common.cu
    src/creatures/creature_spawner.cu
    src/creatures/creature_movement.cu
    src/creatures/creature_interaction.cu
    src/creatures/creature_lifecycle.cu
    src/creatures/stream_compaction.cu
    src/creatures/creature_renderer.cu
    src/creatures/creature_manager.cu
)
set(ECOSYSTEM_SOURCES
    src/ecosystem/ecosystem_common.cu
    src/ecosystem/food_web.cu
    src/ecosystem/resource_manager.cu
    src/ecosystem/disease_model.cu
    src/ecosystem/population_control.cu
    src/ecosystem/ecosystem_manager.cu
)
set(ANALYTICS_SOURCES
    src/analytics/analytics_common.cu
    src/analytics/parallel_statistics.cu
    src/analytics/histogram_kernel.cu
    src/analytics/diversity_index.cu
    src/analytics/phylo_visualizer.cu
    src/analytics/heatmap_generator.cu
    src/analytics/data_exporter.cu
    src/analytics/analytics_manager.cu
)
set(RENDERING_SOURCES
    src/rendering/render_common.cu
    src/rendering/atmosphere_renderer.cu
    src/rendering/cloud_raymarcher.cu
    src/rendering/day_night_cycle.cu
    src/rendering/post_processing.cu
    src/rendering/camera_controller.cpp
    src/rendering/imgui_dashboard.cpp
    src/rendering/render_manager.cu
)
set(CHECKPOINT_SOURCES src/checkpoint/checkpoint_system.cu)
set(SCENARIO_SOURCES src/scenario/scenario_loader.cpp)
set(BENCHMARK_SOURCES src/benchmark/benchmark_suite.cu)

set(ALL_SOURCES
    ${CORE_SOURCES} ${SPATIAL_SOURCES} ${TERRAIN_SOURCES} ${FLUID_SOURCES}
    ${CLIMATE_SOURCES} ${GENETICS_SOURCES} ${NEURAL_SOURCES} ${CREATURE_SOURCES}
    ${ECOSYSTEM_SOURCES} ${ANALYTICS_SOURCES} ${RENDERING_SOURCES}
    ${CHECKPOINT_SOURCES} ${SCENARIO_SOURCES} ${BENCHMARK_SOURCES}
    src/main.cu
)

add_executable(genesis ${ALL_SOURCES})
target_include_directories(genesis PRIVATE ${CMAKE_SOURCE_DIR}/src ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(genesis PRIVATE CUDA::cudart CUDA::curand)

# Separable compilation MUST be ON for the architecture's extern __constant__ to work
set_target_properties(genesis PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

if(MSVC)
    # Use proper CMake lists for flags to prevent escaping errors
    target_compile_options(genesis PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor;-Xcompiler=/wd4819>
        $<$<COMPILE_LANGUAGE:CXX>:/EHsc;/Zc:preprocessor;/wd4819>
    )
    target_compile_definitions(genesis PRIVATE CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING _CRT_SECURE_NO_WARNINGS)
endif()

# ============ TEST TARGETS ============
macro(add_genesis_test test_name test_src)
    add_executable(${test_name} ${test_src} ${CORE_SOURCES} ${ARGN})
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${test_name} PRIVATE CUDA::cudart CUDA::curand)
    set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    if(MSVC)
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-Xcompiler=/EHsc;-Xcompiler=/Zc:preprocessor;-Xcompiler=/wd4819>
            $<$<COMPILE_LANGUAGE:CXX>:/EHsc;/Zc:preprocessor;/wd4819>
        )
        target_compile_definitions(${test_name} PRIVATE CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING _CRT_SECURE_NO_WARNINGS)
    endif()
endmacro()

add_genesis_test(test_spatial_hash tests/test_spatial_hash.cu ${SPATIAL_SOURCES})
add_genesis_test(test_sph tests/test_sph.cu)
add_genesis_test(test_genetics tests/test_genetics.cu ${GENETICS_SOURCES})
add_genesis_test(test_neural tests/test_neural.cu)
add_genesis_test(test_ecosystem tests/test_ecosystem.cu ${ECOSYSTEM_SOURCES})
add_genesis_test(test_climate tests/test_climate.cu)
add_genesis_test(test_analytics tests/test_analytics.cu ${ANALYTICS_SOURCES})