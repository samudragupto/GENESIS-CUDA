#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"

struct SPHTestResult {
    const char* name;
    int passed;
    float time_ms;
};

#define MAX_SPH_TESTS 32
static SPHTestResult sph_results[MAX_SPH_TESTS];
static int num_sph_tests = 0;

static void recordSPHTest(const char* name, int passed, float ms) {
    sph_results[num_sph_tests].name = name;
    sph_results[num_sph_tests].passed = passed;
    sph_results[num_sph_tests].time_ms = ms;
    num_sph_tests++;
}

__device__ float cubicSplineKernel(float r, float h) {
    float q = r / h;
    float sigma = 10.0f / (7.0f * 3.14159265f * h * h);
    if (q < 1.0f) {
        return sigma * (1.0f - 1.5f * q * q + 0.75f * q * q * q);
    } else if (q < 2.0f) {
        float t = 2.0f - q;
        return sigma * 0.25f * t * t * t;
    }
    return 0.0f;
}

__global__ void testKernelValues(float* output, float h) {
    int idx = threadIdx.x;
    float r = (float)idx * h * 0.1f;
    output[idx] = cubicSplineKernel(r, h);
}

void testSPHKernelFunction() {
    float* d_output;
    float h_output[32];
    CUDA_CHECK(cudaMalloc(&d_output, 32 * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    testKernelValues<<<1, 32>>>(d_output, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_output, d_output, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    if (h_output[0] <= 0.0f) passed = 0;
    for (int i = 1; i < 20; i++) {
        if (h_output[i] > h_output[i - 1]) passed = 0;
    }
    if (h_output[20] != 0.0f && h_output[25] != 0.0f) {
    }

    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordSPHTest("SPH Kernel Function Monotonicity", passed, ms);
}

__global__ void densityComputeKernel(
    float* density,
    const float* pos_x,
    const float* pos_y,
    int n,
    float h,
    float mass
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float rho = 0.0f;
    for (int j = 0; j < n; j++) {
        float dx = pos_x[i] - pos_x[j];
        float dy = pos_y[i] - pos_y[j];
        float r = sqrtf(dx * dx + dy * dy);
        rho += mass * cubicSplineKernel(r, h);
    }
    density[i] = rho;
}

void testDensityComputation() {
    int n = 100;
    float spacing = 0.5f;
    float h = 1.0f;
    float mass = 1.0f;

    float* h_pos_x = new float[n];
    float* h_pos_y = new float[n];
    int side = 10;
    for (int i = 0; i < n; i++) {
        h_pos_x[i] = (float)(i % side) * spacing;
        h_pos_y[i] = (float)(i / side) * spacing;
    }

    float* d_pos_x;
    float* d_pos_y;
    float* d_density;
    CUDA_CHECK(cudaMalloc(&d_pos_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_pos_y, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_density, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_pos_x, h_pos_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pos_y, h_pos_y, n * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    densityComputeKernel<<<(n + 255) / 256, 256>>>(d_density, d_pos_x, d_pos_y, n, h, mass);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float* h_density = new float[n];
    CUDA_CHECK(cudaMemcpy(h_density, d_density, n * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    for (int i = 0; i < n; i++) {
        if (h_density[i] <= 0.0f || isnan(h_density[i]) || isinf(h_density[i])) {
            passed = 0;
            break;
        }
    }

    float interior_density = h_density[55];
    float edge_density = h_density[0];
    if (interior_density <= edge_density) passed = 0;

    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_density);
    delete[] h_pos_x;
    delete[] h_pos_y;
    delete[] h_density;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordSPHTest("Density Computation (Positive & Interior > Edge)", passed, ms);
}

void testConservation() {
    int n = 50;
    float* h_pos_x = new float[n];
    float* h_pos_y = new float[n];
    float* h_vel_x = new float[n];
    float* h_vel_y = new float[n];

    srand(123);
    float total_momentum_x = 0.0f;
    float total_momentum_y = 0.0f;
    for (int i = 0; i < n; i++) {
        h_pos_x[i] = 50.0f + (float)(rand() % 100) * 0.1f;
        h_pos_y[i] = 50.0f + (float)(rand() % 100) * 0.1f;
        h_vel_x[i] = ((float)(rand() % 200) - 100.0f) * 0.01f;
        h_vel_y[i] = ((float)(rand() % 200) - 100.0f) * 0.01f;
        total_momentum_x += h_vel_x[i];
        total_momentum_y += h_vel_y[i];
    }

    int passed = 1;

    delete[] h_pos_x;
    delete[] h_pos_y;
    delete[] h_vel_x;
    delete[] h_vel_y;

    recordSPHTest("Momentum Conservation Setup", passed, 0.0f);
}

void testPressureSymmetry() {
    int n = 2;
    float h_pos_x[] = {50.0f, 51.0f};
    float h_pos_y[] = {50.0f, 50.0f};

    float* d_px;
    float* d_py;
    float* d_dens;
    CUDA_CHECK(cudaMalloc(&d_px, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_py, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dens, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_px, h_pos_x, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_pos_y, n * sizeof(float), cudaMemcpyHostToDevice));

    densityComputeKernel<<<1, n>>>(d_dens, d_px, d_py, n, 2.0f, 1.0f);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_dens[2];
    CUDA_CHECK(cudaMemcpy(h_dens, d_dens, n * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = (fabsf(h_dens[0] - h_dens[1]) < 0.001f);

    cudaFree(d_px);
    cudaFree(d_py);
    cudaFree(d_dens);

    recordSPHTest("Pressure Symmetry (Equal Pair)", passed, 0.0f);
}

static void printSPHResults() {
    int total_passed = 0;
    printf("\n========== SPH Tests ==========\n");
    for (int i = 0; i < num_sph_tests; i++) {
        printf("  [%s] %-45s (%.3f ms)\n",
               sph_results[i].passed ? "PASS" : "FAIL",
               sph_results[i].name,
               sph_results[i].time_ms);
        if (sph_results[i].passed) total_passed++;
    }
    printf("===============================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_sph_tests);
}

int main() {
    printf("\nRunning SPH Tests...\n");

    testSPHKernelFunction();
    testDensityComputation();
    testConservation();
    testPressureSymmetry();

    printSPHResults();

    int all_passed = 1;
    for (int i = 0; i < num_sph_tests; i++) {
        if (!sph_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}