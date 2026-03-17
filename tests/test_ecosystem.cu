#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/ecosystem/ecosystem_common.cuh"
#include "../src/ecosystem/resource_manager.cuh"

struct EcoTestResult {
    const char* name;
    int passed;
    float time_ms;
};

#define MAX_ETESTS 32
static EcoTestResult e_results[MAX_ETESTS];
static int num_etests = 0;

static void recordETest(const char* name, int passed, float ms) {
    e_results[num_etests].name = name;
    e_results[num_etests].passed = passed;
    e_results[num_etests].time_ms = ms;
    num_etests++;
}

void testVegetationGrowth() {
    int ws = 256;
    int cells = ws * ws;

    EcosystemGridData grid;
    allocateEcosystemGrid(grid, ws);

    float* d_heightmap;
    float* d_temperature;
    float* d_moisture;
    CUDA_CHECK(cudaMalloc(&d_heightmap, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temperature, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moisture, cells * sizeof(float)));

    float* h_height = new float[cells];
    float* h_temp = new float[cells];
    float* h_moist = new float[cells];
    float* h_veg_before = new float[cells];

    for (int i = 0; i < cells; i++) {
        h_height[i] = 0.5f;
        h_temp[i] = 20.0f;
        h_moist[i] = 0.5f;
        h_veg_before[i] = 0.3f;
    }

    CUDA_CHECK(cudaMemcpy(d_heightmap, h_height, cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_temperature, h_temp, cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moisture, h_moist, cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid.d_vegetation, h_veg_before, cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(grid.d_vegetation_capacity, h_moist, cells * sizeof(float), cudaMemcpyHostToDevice));

    EcosystemParams params;
    params.vegetation_growth_rate = 0.01f;
    params.vegetation_spread_rate = 0.005f;
    params.dead_matter_decay_rate = 0.005f;
    params.nutrient_diffusion_rate = 0.1f;
    params.toxicity_decay_rate = 0.001f;
    params.min_growth_temperature = 5.0f;
    params.max_growth_temperature = 40.0f;
    params.optimal_temperature = 22.0f;
    params.min_growth_moisture = 0.1f;
    params.max_vegetation = 2.0f;
    params.dt = 1.0f;

    launchComputeVegetationCapacity(grid, d_heightmap, d_temperature, d_moisture, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int step = 0; step < 100; step++) {
        launchVegetationGrowth(grid, d_heightmap, d_temperature, d_moisture, params, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_veg_after = new float[cells];
    CUDA_CHECK(cudaMemcpy(h_veg_after, grid.d_vegetation, cells * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    int grew = 0;
    for (int i = 0; i < cells; i++) {
        if (h_veg_after[i] > h_veg_before[i]) grew++;
        if (h_veg_after[i] < 0.0f || h_veg_after[i] > params.max_vegetation + 0.01f) {
            passed = 0;
            break;
        }
        if (isnan(h_veg_after[i]) || isinf(h_veg_after[i])) {
            passed = 0;
            break;
        }
    }

    if (grew == 0) passed = 0;

    freeEcosystemGrid(grid);
    cudaFree(d_heightmap);
    cudaFree(d_temperature);
    cudaFree(d_moisture);
    delete[] h_height;
    delete[] h_temp;
    delete[] h_moist;
    delete[] h_veg_before;
    delete[] h_veg_after;

    recordETest("Vegetation Growth (Positive & Bounded)", passed, 0.0f);
}

void testWaterVegetation() {
    int ws = 128;
    int cells = ws * ws;

    EcosystemGridData grid;
    allocateEcosystemGrid(grid, ws);

    float* d_heightmap;
    float* d_temperature;
    float* d_moisture;
    CUDA_CHECK(cudaMalloc(&d_heightmap, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_temperature, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moisture, cells * sizeof(float)));

    float* h_height = new float[cells];
    for (int i = 0; i < cells; i++) h_height[i] = 0.1f;
    CUDA_CHECK(cudaMemcpy(d_heightmap, h_height, cells * sizeof(float), cudaMemcpyHostToDevice));

    float* h_temp = new float[cells];
    float* h_moist = new float[cells];
    for (int i = 0; i < cells; i++) { h_temp[i] = 20.0f; h_moist[i] = 0.5f; }
    CUDA_CHECK(cudaMemcpy(d_temperature, h_temp, cells * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moisture, h_moist, cells * sizeof(float), cudaMemcpyHostToDevice));

    float* h_veg = new float[cells];
    for (int i = 0; i < cells; i++) h_veg[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(grid.d_vegetation, h_veg, cells * sizeof(float), cudaMemcpyHostToDevice));

    EcosystemParams params;
    params.vegetation_growth_rate = 0.01f;
    params.vegetation_spread_rate = 0.005f;
    params.dead_matter_decay_rate = 0.005f;
    params.nutrient_diffusion_rate = 0.1f;
    params.toxicity_decay_rate = 0.001f;
    params.min_growth_temperature = 5.0f;
    params.max_growth_temperature = 40.0f;
    params.optimal_temperature = 22.0f;
    params.min_growth_moisture = 0.1f;
    params.max_vegetation = 2.0f;
    params.dt = 1.0f;

    launchVegetationGrowth(grid, d_heightmap, d_temperature, d_moisture, params, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_veg, grid.d_vegetation, cells * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    for (int i = 0; i < cells; i++) {
        if (h_veg[i] > 0.001f) {
            passed = 0;
            break;
        }
    }

    freeEcosystemGrid(grid);
    cudaFree(d_heightmap);
    cudaFree(d_temperature);
    cudaFree(d_moisture);
    delete[] h_height;
    delete[] h_temp;
    delete[] h_moist;
    delete[] h_veg;

    recordETest("No Vegetation Underwater", passed, 0.0f);
}

void testNutrientDiffusion() {
    int ws = 64;
    int cells = ws * ws;

    EcosystemGridData grid;
    allocateEcosystemGrid(grid, ws);

    float* h_nutrients = new float[cells];
    memset(h_nutrients, 0, cells * sizeof(float));
    h_nutrients[32 * ws + 32] = 100.0f;

    CUDA_CHECK(cudaMemcpy(grid.d_mineral_nutrients, h_nutrients,
        cells * sizeof(float), cudaMemcpyHostToDevice));

    EcosystemParams params;
    params.nutrient_diffusion_rate = 0.2f;
    params.dt = 1.0f;

    for (int step = 0; step < 50; step++) {
        launchNutrientDiffusion(grid, params, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_nutrients, grid.d_mineral_nutrients,
        cells * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    float center = h_nutrients[32 * ws + 32];
    float neighbor = h_nutrients[32 * ws + 33];

    if (center >= 100.0f) passed = 0;
    if (neighbor <= 0.0f) passed = 0;

    for (int i = 0; i < cells; i++) {
        if (h_nutrients[i] < 0.0f || isnan(h_nutrients[i])) {
            passed = 0;
            break;
        }
    }

    freeEcosystemGrid(grid);
    delete[] h_nutrients;

    recordETest("Nutrient Diffusion (Spread & Non-negative)", passed, 0.0f);
}

static void printEResults() {
    int total_passed = 0;
    printf("\n========== Ecosystem Tests ==========\n");
    for (int i = 0; i < num_etests; i++) {
        printf("  [%s] %-45s (%.3f ms)\n",
               e_results[i].passed ? "PASS" : "FAIL",
               e_results[i].name,
               e_results[i].time_ms);
        if (e_results[i].passed) total_passed++;
    }
    printf("=====================================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_etests);
}

int main() {
    printf("\nRunning Ecosystem Tests...\n");

    testVegetationGrowth();
    testWaterVegetation();
    testNutrientDiffusion();

    printEResults();

    int all_passed = 1;
    for (int i = 0; i < num_etests; i++) {
        if (!e_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}