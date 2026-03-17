#ifndef NEURAL_MANAGER_CUH
#define NEURAL_MANAGER_CUH

#include <cuda_runtime.h>
#include "neural_common.cuh"
#include "neural_forward.cuh"
#include "sensory_input.cuh"
#include "decision_extractor.cuh"
#include "genome_to_weights.cuh"
#include "../creatures/creature_common.cuh"

class NeuralManager {
public:
    NeuralBuffers buffers;
    CreatureActions actions;

    cudaStream_t stream_sensory;
    cudaStream_t stream_forward;
    cudaStream_t stream_decision;
    cudaEvent_t event_sensory_done;
    cudaEvent_t event_forward_done;

    int max_creatures;

    void init(int max_c);
    void destroy();

    void runForwardPass(
        const CreatureData& creatures,
        const float* d_heightmap,
        const float* d_vegetation,
        int world_size,
        int num_creatures,
        cudaStream_t stream = 0
    );
};

#endif