#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"

struct ClimateTestResult {
    const char* name;
    int passed;
};

#define MAX_CTESTS 16
static ClimateTestResult c_results[MAX_CTESTS];
static int num_ctests = 0;

static void recordCTest(const char* name, int passed) {
    c_results[num_ctests].name = name;
    c_results[num_ctests].passed = passed;
    num_ctests++;
}

__global__ void heatDiffusionTestKernel(
    float* __restrict__ temp,
    int ws,
    float diffusion,
    float dt
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < 1 || x >= ws - 1 || y < 1 || y >= ws - 1) return;

    int idx = y * ws + x;
    float c = temp[idx];
    float n = temp[(y-1)*ws+x];
    float s = temp[(y+1)*ws+x];
    float e = temp[y*ws+(x+1)];
    float w = temp[y*ws+(x-1)];

    float laplacian = (n + s + e + w - 4.0f * c);
    temp[idx] = c + diffusion * laplacian * dt;
}

void testHeatDiffusion() {
    int ws = 64;
    int cells = ws * ws;
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, cells * sizeof(float)));

    float* h_temp = new float[cells];
    for (int i = 0; i < cells; i++) h_temp[i] = 20.0f;
    h_temp[32 * ws + 32] = 100.0f;

    CUDA_CHECK(cudaMemcpy(d_temp, h_temp, cells * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((ws + 15) / 16, (ws + 15) / 16);

    for (int step = 0; step < 100; step++) {
        heatDiffusionTestKernel<<<grid, block>>>(d_temp, ws, 0.2f, 0.1f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, cells * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    float center = h_temp[32 * ws + 32];
    if (center >= 100.0f) passed = 0;
    if (center < 20.0f) passed = 0;

    float far_corner = h_temp[0];
    if (far_corner <= 20.0f && center > 25.0f) {
    } else if (far_corner < 19.0f) {
        passed = 0;
    }

    for (int i = 0; i < cells; i++) {
        if (isnan(h_temp[i]) || isinf(h_temp[i])) { passed = 0; break; }
    }

    cudaFree(d_temp);
    delete[] h_temp;

    recordCTest("Heat Diffusion Convergence", passed);
}

void testTemperatureRange() {
    int ws = 128;
    int cells = ws * ws;
    float* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, cells * sizeof(float)));

    float* h_temp = new float[cells];
    for (int y = 0; y < ws; y++) {
        for (int x = 0; x < ws; x++) {
            float lat = (float)y / (float)ws;
            h_temp[y * ws + x] = 35.0f - 40.0f * fabsf(lat - 0.5f);
        }
    }

    CUDA_CHECK(cudaMemcpy(d_temp, h_temp, cells * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid((ws + 15) / 16, (ws + 15) / 16);

    for (int step = 0; step < 50; step++) {
        heatDiffusionTestKernel<<<grid, block>>>(d_temp, ws, 0.1f, 0.1f);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_temp, d_temp, cells * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    float equator_temp = h_temp[(ws / 2) * ws + ws / 2];
    float pole_temp = h_temp[0];

    if (equator_temp <= pole_temp) passed = 0;

    for (int i = 0; i < cells; i++) {
        if (h_temp[i] < -100.0f || h_temp[i] > 200.0f) { passed = 0; break; }
    }

    cudaFree(d_temp);
    delete[] h_temp;

    recordCTest("Temperature Gradient Equator > Poles", passed);
}

static void printCResults() {
    int total_passed = 0;
    printf("\n========== Climate Tests ==========\n");
    for (int i = 0; i < num_ctests; i++) {
        printf("  [%s] %-45s\n",
               c_results[i].passed ? "PASS" : "FAIL",
               c_results[i].name);
        if (c_results[i].passed) total_passed++;
    }
    printf("===================================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_ctests);
}

int main() {
    printf("\nRunning Climate Tests...\n");

    testHeatDiffusion();
    testTemperatureRange();

    printCResults();

    int all_passed = 1;
    for (int i = 0; i < num_ctests; i++) {
        if (!c_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}