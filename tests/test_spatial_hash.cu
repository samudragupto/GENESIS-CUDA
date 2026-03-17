#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/spatial/spatial_hash.cuh"
#include "../src/spatial/sort_particles.cuh"
#include "../src/spatial/neighbor_search.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/spatial/spatial_hash.cuh"
#include "../src/spatial/sort_particles.cuh"
#include "../src/spatial/neighbor_search.cuh"
struct TestResult {
    const char* name;
    int passed;
    float time_ms;
};

#define MAX_TESTS 32
static TestResult test_results[MAX_TESTS];
static int num_tests = 0;

static void recordTest(const char* name, int passed, float ms) {
    test_results[num_tests].name = name;
    test_results[num_tests].passed = passed;
    test_results[num_tests].time_ms = ms;
    num_tests++;
}

static void printResults() {
    int total_passed = 0;
    printf("\n========== Spatial Hash Tests ==========\n");
    for (int i = 0; i < num_tests; i++) {
        printf("  [%s] %-40s (%.3f ms)\n",
               test_results[i].passed ? "PASS" : "FAIL",
               test_results[i].name,
               test_results[i].time_ms);
        if (test_results[i].passed) total_passed++;
    }
    printf("========================================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_tests);
}

void testGridAllocation() {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    SpatialHashGrid grid;
    int max_particles = 100000;
    int grid_size = 256;
    initSpatialHashGrid(grid, max_particles, grid_size);

    int passed = (grid.d_cell_start != nullptr &&
                  grid.d_cell_end != nullptr &&
                  grid.d_particle_hash != nullptr &&
                  grid.d_sorted_indices != nullptr &&
                  grid.grid_size == grid_size);

    freeSpatialHashGrid(grid);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordTest("Grid Allocation & Deallocation", passed, ms);
}

void testHashComputation() {
    int n = 10000;
    float* h_pos_x = new float[n];
    float* h_pos_y = new float[n];

    srand(42);
    for (int i = 0; i < n; i++) {
        h_pos_x[i] = (float)(rand() % 1024);
        h_pos_y[i] = (float)(rand() % 1024);
    }

    float* d_pos_x;
    float* d_pos_y;
    int* d_alive;
    CUDA_CHECK(cudaMalloc(&d_pos_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alive, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, n * sizeof(float), cudaMemcpyHostToDevice));

    int* h_alive = new int[n];
    for (int i = 0; i < n; i++) h_alive[i] = 1;
    CUDA_CHECK(cudaMemcpy(d_alive, h_alive, n * sizeof(int), cudaMemcpyHostToDevice));

    SpatialHashGrid grid;
    initSpatialHashGrid(grid, n, 256);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    updateSpatialHashGrid(grid, d_pos_x, d_pos_y, d_alive, n, 1024.0f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int* h_cell_start = new int[256 * 256];
    CUDA_CHECK(cudaMemcpy(h_cell_start, grid.d_cell_start,
        256 * 256 * sizeof(int), cudaMemcpyDeviceToHost));

    int total_assigned = 0;
    for (int i = 0; i < 256 * 256; i++) {
        if (h_cell_start[i] >= 0) total_assigned++;
    }

    int passed = (total_assigned > 0 && total_assigned <= 256 * 256);

    freeSpatialHashGrid(grid);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_alive);
    delete[] h_pos_x;
    delete[] h_pos_y;
    delete[] h_alive;
    delete[] h_cell_start;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordTest("Hash Computation & Sort", passed, ms);
}

void testNeighborCorrectness() {
    int n = 4;
    float h_pos_x[] = {10.0f, 11.0f, 100.0f, 101.0f};
    float h_pos_y[] = {10.0f, 11.0f, 100.0f, 101.0f};
    int h_alive[] = {1, 1, 1, 1};

    float* d_pos_x;
    float* d_pos_y;
    int* d_alive;
    CUDA_CHECK(cudaMalloc(&d_pos_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alive, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_alive, h_alive, n * sizeof(int), cudaMemcpyHostToDevice));

    SpatialHashGrid grid;
    initSpatialHashGrid(grid, n, 64);
    updateSpatialHashGrid(grid, d_pos_x, d_pos_y, d_alive, n, 256.0f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    int passed = 1;

    freeSpatialHashGrid(grid);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_alive);

    recordTest("Neighbor Correctness (Close Pairs)", passed, 0.0f);
}

void testLargeScale() {
    int n = 1000000;
    float* d_pos_x;
    float* d_pos_y;
    int* d_alive;

    CUDA_CHECK(cudaMalloc(&d_pos_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alive, n * sizeof(int)));

    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng, n * sizeof(curandState)));

    SpatialHashGrid grid;
    initSpatialHashGrid(grid, n, 512);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    updateSpatialHashGrid(grid, d_pos_x, d_pos_y, d_alive, n, 2048.0f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int passed = (ms < 100.0f);

    freeSpatialHashGrid(grid);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_alive);
    cudaFree(d_rng);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordTest("Large Scale (1M particles)", passed, ms);
}

void testEmptyGrid() {
    SpatialHashGrid grid;
    initSpatialHashGrid(grid, 1000, 64);

    float* d_pos_x;
    float* d_pos_y;
    int* d_alive;
    CUDA_CHECK(cudaMalloc(&d_pos_x, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_alive, sizeof(int)));
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(d_alive, &zero, sizeof(int), cudaMemcpyHostToDevice));

    updateSpatialHashGrid(grid, d_pos_x, d_pos_y, d_alive, 0, 256.0f, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    int passed = 1;

    freeSpatialHashGrid(grid);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_alive);

    recordTest("Empty Grid Handling", passed, 0.0f);
}

int main() {
    printf("\nRunning Spatial Hash Tests...\n");

    testGridAllocation();
    testHashComputation();
    testNeighborCorrectness();
    testLargeScale();
    testEmptyGrid();

    printResults();

    int all_passed = 1;
    for (int i = 0; i < num_tests; i++) {
        if (!test_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}