#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/analytics/diversity_index.cuh"

struct AnalyticsTestResult {
    const char* name;
    int passed;
};

#define MAX_ATESTS 16
static AnalyticsTestResult a_results[MAX_ATESTS];
static int num_atests = 0;

static void recordATest(const char* name, int passed) {
    a_results[num_atests].name = name;
    a_results[num_atests].passed = passed;
    num_atests++;
}

void testShannonUniform() {
    int max_species = 10;
    int* d_pop;
    CUDA_CHECK(cudaMalloc(&d_pop, max_species * sizeof(int)));

    int h_pop[10] = {100, 100, 100, 100, 100, 100, 100, 100, 100, 100};
    CUDA_CHECK(cudaMemcpy(d_pop, h_pop, max_species * sizeof(int), cudaMemcpyHostToDevice));

    DiversityResult result;
    allocateDiversityResult(result);

    launchShannonDiversity(d_pop, max_species, 1000, result, 0);
    launchSimpsonDiversity(d_pop, max_species, 1000, result, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float shannon;
    CUDA_CHECK(cudaMemcpy(&shannon, result.d_shannon, sizeof(float), cudaMemcpyDeviceToHost));

    float expected = logf(10.0f);
    int passed = (fabsf(shannon - expected) < 0.01f);

    freeDiversityResult(result);
    cudaFree(d_pop);

    recordATest("Shannon Diversity (Uniform = ln(S))", passed);
}

void testShannonSingle() {
    int max_species = 10;
    int* d_pop;
    CUDA_CHECK(cudaMalloc(&d_pop, max_species * sizeof(int)));

    int h_pop[10] = {1000, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpy(d_pop, h_pop, max_species * sizeof(int), cudaMemcpyHostToDevice));

    DiversityResult result;
    allocateDiversityResult(result);

    launchShannonDiversity(d_pop, max_species, 1000, result, 0);
    launchSimpsonDiversity(d_pop, max_species, 1000, result, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float shannon;
    CUDA_CHECK(cudaMemcpy(&shannon, result.d_shannon, sizeof(float), cudaMemcpyDeviceToHost));

    int passed = (fabsf(shannon) < 0.01f);

    freeDiversityResult(result);
    cudaFree(d_pop);

    recordATest("Shannon Diversity (Single species = 0)", passed);
}

void testSimpsonDominance() {
    int max_species = 5;
    int* d_pop;
    CUDA_CHECK(cudaMalloc(&d_pop, max_species * sizeof(int)));

    int h_pop_dom[5] = {990, 1, 1, 1, 1};
    CUDA_CHECK(cudaMemcpy(d_pop, h_pop_dom, max_species * sizeof(int), cudaMemcpyHostToDevice));

    DiversityResult result;
    allocateDiversityResult(result);

    launchShannonDiversity(d_pop, max_species, 994, result, 0);
    launchSimpsonDiversity(d_pop, max_species, 994, result, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float simpson_dom;
    CUDA_CHECK(cudaMemcpy(&simpson_dom, result.d_simpson, sizeof(float), cudaMemcpyDeviceToHost));

    float zero_f = 0.0f;
    CUDA_CHECK(cudaMemcpy(result.d_shannon, &zero_f, sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(result.d_simpson, &zero_f, sizeof(float), cudaMemcpyHostToDevice));

    int h_pop_even[5] = {200, 200, 200, 200, 194};
    CUDA_CHECK(cudaMemcpy(d_pop, h_pop_even, max_species * sizeof(int), cudaMemcpyHostToDevice));

    launchShannonDiversity(d_pop, max_species, 994, result, 0);
    launchSimpsonDiversity(d_pop, max_species, 994, result, 0);
    CUDA_CHECK(cudaDeviceSynchronize());

    float simpson_even;
    CUDA_CHECK(cudaMemcpy(&simpson_even, result.d_simpson, sizeof(float), cudaMemcpyDeviceToHost));

    int passed = (simpson_even > simpson_dom);

    freeDiversityResult(result);
    cudaFree(d_pop);

    recordATest("Simpson: Even > Dominant", passed);
}

static void printAResults() {
    int total_passed = 0;
    printf("\n========== Analytics Tests ==========\n");
    for (int i = 0; i < num_atests; i++) {
        printf("  [%s] %-45s\n",
               a_results[i].passed ? "PASS" : "FAIL",
               a_results[i].name);
        if (a_results[i].passed) total_passed++;
    }
    printf("=====================================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_atests);
}

int main() {
    printf("\nRunning Analytics Tests...\n");

    testShannonUniform();
    testShannonSingle();
    testSimpsonDominance();

    printAResults();

    int all_passed = 1;
    for (int i = 0; i < num_atests; i++) {
        if (!a_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}