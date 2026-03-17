#ifndef NEURAL_COMMON_CUH
#define NEURAL_COMMON_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct CreatureActions {
    float* d_move_dx;
    float* d_move_dy;
    float* d_speed;
    float* d_eat;
    float* d_attack;
    float* d_reproduce;
    float* d_flee;
    float* d_social_signal;
    int    max_creatures;
};

struct NeuralBuffers {
    float* d_input_buffer;
    float* d_hidden1_buffer;
    float* d_hidden2_buffer;
    float* d_output_buffer;
    float* d_sensory_input;
    float* d_neural_output;
    float* d_neural_weights;
    int    max_creatures;
};

struct WorldDataPtrs {
    const float* d_heightmap;
    const float* d_vegetation;
    const float* d_temperature;
    const float* d_moisture;
    int world_size;
};

struct SpatialGridPtrs {
    const int* d_cell_start;
    const int* d_cell_end;
    const int* d_sorted_indices;
    int grid_size;
    float cell_size;
};

struct NeuralLayerConfig {
    int input_size;
    int output_size;
    int weight_offset;
    int activation_type;
};

void allocateCreatureActions(CreatureActions& actions, int max_creatures);
void freeCreatureActions(CreatureActions& actions);
void allocateNeuralBuffers(NeuralBuffers& buffers, int max_creatures);
void freeNeuralBuffers(NeuralBuffers& buffers);

#endif