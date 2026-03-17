#include "neural_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void NeuralManager::init(int max_c) {
    max_creatures = max_c;

    CUDA_CHECK(cudaMalloc(&buffers.d_sensory_input,
        (size_t)max_c * NEURAL_INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_neural_output,
        (size_t)max_c * NEURAL_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers.d_neural_weights,
        (size_t)max_c * NEURAL_WEIGHT_COUNT * sizeof(float)));

    CUDA_CHECK(cudaMemset(buffers.d_sensory_input, 0,
        (size_t)max_c * NEURAL_INPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(buffers.d_neural_output, 0,
        (size_t)max_c * NEURAL_OUTPUT_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemset(buffers.d_neural_weights, 0,
        (size_t)max_c * NEURAL_WEIGHT_COUNT * sizeof(float)));

    buffers.d_input_buffer = buffers.d_sensory_input;
    buffers.d_output_buffer = buffers.d_neural_output;
    buffers.d_hidden1_buffer = nullptr;
    buffers.d_hidden2_buffer = nullptr;
    buffers.max_creatures = max_c;

    allocateCreatureActions(actions, max_c);

    CUDA_CHECK(cudaStreamCreate(&stream_sensory));
    CUDA_CHECK(cudaStreamCreate(&stream_forward));
    CUDA_CHECK(cudaStreamCreate(&stream_decision));
    CUDA_CHECK(cudaEventCreate(&event_sensory_done));
    CUDA_CHECK(cudaEventCreate(&event_forward_done));
}

void NeuralManager::destroy() {
    cudaFree(buffers.d_sensory_input);
    cudaFree(buffers.d_neural_output);
    cudaFree(buffers.d_neural_weights);

    freeCreatureActions(actions);

    cudaStreamDestroy(stream_sensory);
    cudaStreamDestroy(stream_forward);
    cudaStreamDestroy(stream_decision);
    cudaEventDestroy(event_sensory_done);
    cudaEventDestroy(event_forward_done);
}

void NeuralManager::runForwardPass(
    const CreatureData& creatures,
    const float* d_heightmap,
    const float* d_vegetation,
    int world_size,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    SensoryContext ctx;
    ctx.creatures = &creatures;
    ctx.world.d_heightmap = d_heightmap;
    ctx.world.d_vegetation = d_vegetation;
    ctx.world.d_temperature = nullptr;
    ctx.world.d_moisture = nullptr;
    ctx.world.world_size = world_size;
    ctx.grid.d_cell_start = nullptr;
    ctx.grid.d_cell_end = nullptr;
    ctx.grid.d_sorted_indices = nullptr;
    ctx.grid.grid_size = 0;
    ctx.grid.cell_size = 4.0f;
    ctx.num_creatures = num_creatures;

    launchSensoryInputKernel(buffers.d_sensory_input, ctx, stream_sensory);
    CUDA_CHECK(cudaEventRecord(event_sensory_done, stream_sensory));

    CUDA_CHECK(cudaStreamWaitEvent(stream_forward, event_sensory_done, 0));

    launchNeuralForwardPass(
        buffers.d_sensory_input,
        creatures.d_neural_weights,
        creatures.d_genomes,
        buffers.d_neural_output,
        num_creatures,
        GENOME_SIZE,
        NEURAL_WEIGHT_COUNT,
        stream_forward
    );
    CUDA_CHECK(cudaEventRecord(event_forward_done, stream_forward));

    CUDA_CHECK(cudaStreamWaitEvent(stream_decision, event_forward_done, 0));

    launchDecisionExtraction(
        buffers.d_neural_output,
        actions,
        num_creatures,
        stream_decision
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_decision));
}