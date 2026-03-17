#ifndef POST_PROCESSING_CUH
#define POST_PROCESSING_CUH

#include <cuda_runtime.h>
#include "render_common.cuh"

struct PostProcessParams {
    float bloom_threshold;
    float bloom_intensity;
    int   bloom_blur_passes;
    float exposure;
    float gamma;
    float saturation;
    float vignette_strength;
    float vignette_radius;
    int   fxaa_enabled;
};

struct PostProcessBuffers {
    float4* d_bright_pass;
    float4* d_bloom_buffer;
    float4* d_temp_buffer;
    int width;
    int height;
};

void allocatePostProcessBuffers(PostProcessBuffers& pp, int width, int height);
void freePostProcessBuffers(PostProcessBuffers& pp);

void launchBrightPass(
    const FrameBuffer& fb,
    PostProcessBuffers& pp,
    float threshold,
    cudaStream_t stream = 0
);

void launchBloomBlur(
    PostProcessBuffers& pp,
    int passes,
    cudaStream_t stream = 0
);

void launchBloomComposite(
    FrameBuffer& fb,
    const PostProcessBuffers& pp,
    float intensity,
    cudaStream_t stream = 0
);

void launchToneMapping(
    FrameBuffer& fb,
    float exposure,
    float gamma,
    cudaStream_t stream = 0
);

void launchFXAA(
    FrameBuffer& fb,
    PostProcessBuffers& pp,
    cudaStream_t stream = 0
);

void launchVignette(
    FrameBuffer& fb,
    float strength,
    float radius,
    cudaStream_t stream = 0
);

void launchFullPostProcess(
    FrameBuffer& fb,
    PostProcessBuffers& pp,
    const PostProcessParams& params,
    cudaStream_t stream = 0
);

#endif