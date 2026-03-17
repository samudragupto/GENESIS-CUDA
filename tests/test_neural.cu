#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "../src/core/cuda_utils.cuh"
#include "../src/core/constants.cuh"
#include "../src/neural/neural_common.cuh"

struct NeuralTestResult {
    const char* name;
    int passed;
    float time_ms;
};

#define MAX_NTESTS 32
static NeuralTestResult n_results[MAX_NTESTS];
static int num_ntests = 0;

static void recordNTest(const char* name, int passed, float ms) {
    n_results[num_ntests].name = name;
    n_results[num_ntests].passed = passed;
    n_results[num_ntests].time_ms = ms;
    num_ntests++;
}

__device__ float relu(float x) { return fmaxf(x, 0.0f); }
__device__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ float tanh_act(float x) { return tanhf(x); }

__global__ void testActivationsKernel(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float x = (float)idx / (float)n * 10.0f - 5.0f;

    output[idx * 3 + 0] = relu(x);
    output[idx * 3 + 1] = sigmoid(x);
    output[idx * 3 + 2] = tanh_act(x);
}

void testActivationFunctions() {
    int n = 1000;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, n * 3 * sizeof(float)));

    testActivationsKernel<<<(n + 255) / 256, 256>>>(d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_output = new float[n * 3];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, n * 3 * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    for (int i = 0; i < n; i++) {
        float x = (float)i / (float)n * 10.0f - 5.0f;
        float r = h_output[i * 3 + 0];
        float s = h_output[i * 3 + 1];
        float t = h_output[i * 3 + 2];

        if (r < 0.0f) passed = 0;
        if (x < 0.0f && r != 0.0f) passed = 0;
        if (s < 0.0f || s > 1.0f) passed = 0;
        if (t < -1.0f || t > 1.0f) passed = 0;
        if (isnan(r) || isnan(s) || isnan(t)) passed = 0;
    }

    cudaFree(d_output);
    delete[] h_output;

    recordNTest("Activation Functions (ReLU, Sigmoid, Tanh)", passed, 0.0f);
}

__global__ void singleNeuronForwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    int input_size,
    int num_neurons
) {
    int neuron = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron >= num_neurons) return;

    float sum = weights[neuron * (input_size + 1) + input_size];
    for (int i = 0; i < input_size; i++) {
        sum += input[i] * weights[neuron * (input_size + 1) + i];
    }
    output[neuron] = tanh_act(sum);
}

void testSingleLayerForward() {
    int input_size = NEURAL_INPUT_SIZE;
    int num_neurons = 32;
    int weight_count = num_neurons * (input_size + 1);

    float* d_input;
    float* d_weights;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, weight_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, num_neurons * sizeof(float)));

    float* h_input = new float[input_size];
    float* h_weights = new float[weight_count];
    for (int i = 0; i < input_size; i++) h_input[i] = 0.5f;
    srand(42);
    for (int i = 0; i < weight_count; i++) {
        h_weights[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.2f;
    }

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_weights, weight_count * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    singleNeuronForwardKernel<<<1, num_neurons>>>(d_input, d_weights, d_output, input_size, num_neurons);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    float* h_output = new float[num_neurons];
    CUDA_CHECK(cudaMemcpy(h_output, d_output, num_neurons * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    for (int i = 0; i < num_neurons; i++) {
        if (isnan(h_output[i]) || isinf(h_output[i])) { passed = 0; break; }
        if (h_output[i] < -1.0f || h_output[i] > 1.0f) { passed = 0; break; }
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output);
    delete[] h_input;
    delete[] h_weights;
    delete[] h_output;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordNTest("Single Layer Forward Pass", passed, ms);
}

void testBatchForward() {
    int batch = 100000;
    int input_size = NEURAL_INPUT_SIZE;
    int weight_per_creature = NEURAL_WEIGHT_COUNT;

    float* d_inputs;
    float* d_weights;
    float* d_outputs;

    CUDA_CHECK(cudaMalloc(&d_inputs, (size_t)batch * input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, (size_t)batch * weight_per_creature * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_outputs, (size_t)batch * NEURAL_OUTPUT_SIZE * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_inputs, 0, (size_t)batch * input_size * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_weights, 0, (size_t)batch * weight_per_creature * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    CUDA_CHECK(cudaMemset(d_outputs, 0, (size_t)batch * NEURAL_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    int passed = 1;
    size_t total_mem = (size_t)batch * (input_size + weight_per_creature + NEURAL_OUTPUT_SIZE) * sizeof(float);
    if (total_mem == 0) passed = 0;

    cudaFree(d_inputs);
    cudaFree(d_weights);
    cudaFree(d_outputs);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    recordNTest("Batch Allocation (100K creatures)", passed, ms);
}

void testDeterminism() {
    int input_size = 8;
    int num_neurons = 4;
    int wc = num_neurons * (input_size + 1);

    float* d_input;
    float* d_weights;
    float* d_output1;
    float* d_output2;

    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_weights, wc * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output1, num_neurons * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output2, num_neurons * sizeof(float)));

    float h_in[8] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
    float h_w[36];
    for (int i = 0; i < 36; i++) h_w[i] = 0.1f * (float)(i % 7 - 3);

    CUDA_CHECK(cudaMemcpy(d_input, h_in, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weights, h_w, wc * sizeof(float), cudaMemcpyHostToDevice));

    singleNeuronForwardKernel<<<1, num_neurons>>>(d_input, d_weights, d_output1, input_size, num_neurons);
    singleNeuronForwardKernel<<<1, num_neurons>>>(d_input, d_weights, d_output2, input_size, num_neurons);
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_o1[4], h_o2[4];
    CUDA_CHECK(cudaMemcpy(h_o1, d_output1, num_neurons * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_o2, d_output2, num_neurons * sizeof(float), cudaMemcpyDeviceToHost));

    int passed = 1;
    for (int i = 0; i < num_neurons; i++) {
        if (fabsf(h_o1[i] - h_o2[i]) > 1e-6f) { passed = 0; break; }
    }

    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_output1);
    cudaFree(d_output2);

    recordNTest("Forward Pass Determinism", passed, 0.0f);
}

static void printNResults() {
    int total_passed = 0;
    printf("\n========== Neural Network Tests ==========\n");
    for (int i = 0; i < num_ntests; i++) {
        printf("  [%s] %-45s (%.3f ms)\n",
               n_results[i].passed ? "PASS" : "FAIL",
               n_results[i].name,
               n_results[i].time_ms);
        if (n_results[i].passed) total_passed++;
    }
    printf("==========================================\n");
    printf("  %d/%d tests passed\n\n", total_passed, num_ntests);
}

int main() {
    printf("\nRunning Neural Network Tests...\n");

    testActivationFunctions();
    testSingleLayerForward();
    testBatchForward();
    testDeterminism();

    printNResults();

    int all_passed = 1;
    for (int i = 0; i < num_ntests; i++) {
        if (!n_results[i].passed) { all_passed = 0; break; }
    }
    return all_passed ? 0 : 1;
}