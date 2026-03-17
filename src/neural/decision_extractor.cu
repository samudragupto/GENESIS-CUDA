#include "decision_extractor.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void decisionExtractionKernel(
    const float* __restrict__ neural_output,
    float* __restrict__ move_dx,
    float* __restrict__ move_dy,
    float* __restrict__ speed,
    float* __restrict__ eat,
    float* __restrict__ attack,
    float* __restrict__ reproduce,
    float* __restrict__ flee,
    float* __restrict__ social,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    const float* out = neural_output + idx * NEURAL_OUTPUT_SIZE;

    move_dx[idx] = tanhf(out[0]);
    move_dy[idx] = tanhf(out[1]);
    speed[idx] = 1.0f / (1.0f + expf(-out[2]));
    eat[idx] = 1.0f / (1.0f + expf(-out[3]));
    attack[idx] = 1.0f / (1.0f + expf(-out[4]));
    reproduce[idx] = 1.0f / (1.0f + expf(-out[5]));
    flee[idx] = 1.0f / (1.0f + expf(-out[6]));
    social[idx] = tanhf(out[7]);
}

void launchDecisionExtraction(
    const float* d_neural_output,
    CreatureActions& actions,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    decisionExtractionKernel<<<grid, block, 0, stream>>>(
        d_neural_output,
        actions.d_move_dx, actions.d_move_dy, actions.d_speed,
        actions.d_eat, actions.d_attack,
        actions.d_reproduce, actions.d_flee,
        actions.d_social_signal,
        num_creatures
    );
}

void allocateCreatureActions(CreatureActions& actions, int max_creatures) {
    actions.max_creatures = max_creatures;
    CUDA_CHECK(cudaMalloc(&actions.d_move_dx, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_move_dy, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_speed, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_eat, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_attack, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_reproduce, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_flee, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&actions.d_social_signal, max_creatures * sizeof(float)));

    CUDA_CHECK(cudaMemset(actions.d_move_dx, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_move_dy, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_speed, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_eat, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_attack, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_reproduce, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_flee, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(actions.d_social_signal, 0, max_creatures * sizeof(float)));
}

void freeCreatureActions(CreatureActions& actions) {
    cudaFree(actions.d_move_dx);
    cudaFree(actions.d_move_dy);
    cudaFree(actions.d_speed);
    cudaFree(actions.d_eat);
    cudaFree(actions.d_attack);
    cudaFree(actions.d_reproduce);
    cudaFree(actions.d_flee);
    cudaFree(actions.d_social_signal);
}