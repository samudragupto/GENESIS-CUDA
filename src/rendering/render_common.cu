#include "render_common.cuh"
#include "../core/cuda_utils.cuh"

void allocateFrameBuffer(FrameBuffer& fb, int width, int height) {
    fb.width = width;
    fb.height = height;
    CUDA_CHECK(cudaMalloc(&fb.d_color, width * height * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&fb.d_depth, width * height * sizeof(float)));
}

void freeFrameBuffer(FrameBuffer& fb) {
    cudaFree(fb.d_color);
    cudaFree(fb.d_depth);
}

__global__ void clearFrameBufferKernel(
    float4* __restrict__ color,
    float* __restrict__ depth,
    float4 clear_color,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    color[idx] = clear_color;
    depth[idx] = 1e30f;
}

void clearFrameBuffer(FrameBuffer& fb, float4 clear_color, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((fb.width + 15) / 16, (fb.height + 15) / 16);
    clearFrameBufferKernel<<<grid, block, 0, stream>>>(
        fb.d_color, fb.d_depth, clear_color, fb.width, fb.height
    );
}