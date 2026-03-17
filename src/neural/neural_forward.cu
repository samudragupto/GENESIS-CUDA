#include "neural_forward.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include "activation_functions.cuh"

__global__ void neuralForwardKernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ genomes,
    float* __restrict__ output,
    int num_creatures,
    int genome_size,
    int weight_count
) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int creature_idx = global_thread / WARP_SIZE;
    int lane = global_thread % WARP_SIZE;

    if (creature_idx >= num_creatures) return;

    const float* w = weights + creature_idx * weight_count;
    const float* inp = input + creature_idx * NEURAL_INPUT_SIZE;
    float* out = output + creature_idx * NEURAL_OUTPUT_SIZE;

    int act_type = (int)(genomes[creature_idx * genome_size + GENE_ACTIVATION_TYPE] * 3.99f);

    if (lane < NEURAL_HIDDEN1_SIZE) {
        float sum = w[B1_OFFSET + lane];
        for (int i = 0; i < NEURAL_INPUT_SIZE; i++) {
            sum += inp[i] * w[W1_OFFSET + lane * NEURAL_INPUT_SIZE + i];
        }
        sum = apply_activation(sum, act_type);
        float h1_val = sum;

        if (lane < NEURAL_HIDDEN2_SIZE) {
            float sum2 = w[B2_OFFSET + lane];
            for (int j = 0; j < NEURAL_HIDDEN1_SIZE; j++) {
                float h1j = w[B1_OFFSET + j];
                for (int i = 0; i < NEURAL_INPUT_SIZE; i++) {
                    h1j += inp[i] * w[W1_OFFSET + j * NEURAL_INPUT_SIZE + i];
                }
                h1j = apply_activation(h1j, act_type);
                sum2 += h1j * w[W2_OFFSET + lane * NEURAL_HIDDEN1_SIZE + j];
            }
            sum2 = apply_activation(sum2, act_type);

            if (lane < NEURAL_OUTPUT_SIZE) {
                float sum3 = w[B3_OFFSET + lane];
                for (int k = 0; k < NEURAL_HIDDEN2_SIZE; k++) {
                    float h2k = w[B2_OFFSET + k];
                    for (int j = 0; j < NEURAL_HIDDEN1_SIZE; j++) {
                        float h1j = w[B1_OFFSET + j];
                        for (int i = 0; i < NEURAL_INPUT_SIZE; i++) {
                            h1j += inp[i] * w[W1_OFFSET + j * NEURAL_INPUT_SIZE + i];
                        }
                        h1j = apply_activation(h1j, act_type);
                        h2k += h1j * w[W2_OFFSET + k * NEURAL_HIDDEN1_SIZE + j];
                    }
                    h2k = apply_activation(h2k, act_type);
                    sum3 += h2k * w[W3_OFFSET + lane * NEURAL_HIDDEN2_SIZE + k];
                }
                out[lane] = tanhf(sum3);
            }
        }
    }
}

void launchNeuralForwardPass(
    const float* d_input,
    const float* d_weights,
    const float* d_genomes,
    float* d_output,
    int num_creatures,
    int genome_size,
    int weight_count,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int threads_needed = num_creatures * WARP_SIZE;
    int block = NEURAL_BLOCK_SIZE;
    int grid = (threads_needed + block - 1) / block;

    neuralForwardKernel<<<grid, block, 0, stream>>>(
        d_input, d_weights, d_genomes, d_output,
        num_creatures, genome_size, weight_count
    );
}