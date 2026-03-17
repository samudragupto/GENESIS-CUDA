#ifndef ACTIVATION_FUNCTIONS_CUH
#define ACTIVATION_FUNCTIONS_CUH

#include <cuda_runtime.h>
#include <math.h>

enum ActivationType : int {
    ACT_RELU = 0,
    ACT_TANH = 1,
    ACT_SIGMOID = 2,
    ACT_SWISH = 3
};

__device__ __forceinline__ float act_relu(float x) {
    return fmaxf(0.0f, x);
}

__device__ __forceinline__ float act_tanh(float x) {
    return tanhf(x);
}

__device__ __forceinline__ float act_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float act_swish(float x) {
    return x * (1.0f / (1.0f + expf(-x)));
}

__device__ __forceinline__ float apply_activation(float x, int act_type) {
    switch (act_type) {
        case ACT_RELU:    return act_relu(x);
        case ACT_TANH:    return act_tanh(x);
        case ACT_SIGMOID: return act_sigmoid(x);
        case ACT_SWISH:   return act_swish(x);
        default:          return act_relu(x);
    }
}

__device__ __forceinline__ float output_activation(float x) {
    return tanhf(x);
}

#endif