#ifndef CLOUD_RAYMARCHER_CUH
#define CLOUD_RAYMARCHER_CUH

#include <cuda_runtime.h>
#include "render_common.cuh"

struct CloudParams {
    float cloud_base_height;
    float cloud_top_height;
    float cloud_coverage;
    float cloud_density;
    float cloud_speed;
    float cloud_scale;
    float time;
    int   num_steps;
    int   num_light_steps;
    float light_absorption;
    float ambient_light;
};

void launchCloudRaymarching(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    const CloudParams& clouds,
    cudaStream_t stream = 0
);

#endif