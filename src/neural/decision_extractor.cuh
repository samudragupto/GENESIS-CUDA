#ifndef DECISION_EXTRACTOR_CUH
#define DECISION_EXTRACTOR_CUH

#include <cuda_runtime.h>
#include "neural_common.cuh"

void launchDecisionExtraction(
    const float* d_neural_output,
    CreatureActions& actions,
    int num_creatures,
    cudaStream_t stream = 0
);

void allocateCreatureActions(CreatureActions& actions, int max_creatures);
void freeCreatureActions(CreatureActions& actions);

#endif