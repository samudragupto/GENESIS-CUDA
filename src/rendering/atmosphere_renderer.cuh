#ifndef ATMOSPHERE_RENDERER_CUH
#define ATMOSPHERE_RENDERER_CUH

#include <cuda_runtime.h>
#include "render_common.cuh"

struct AtmosphereParams {
    float planet_radius;
    float atmosphere_radius;
    float rayleigh_scale_height;
    float mie_scale_height;
    float mie_g;
    float3 rayleigh_coeff;
    float mie_coeff;
    int num_samples;
    int num_light_samples;
};

void launchAtmosphericScattering(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    const AtmosphereParams& atmo,
    cudaStream_t stream = 0
);

void launchSkyDome(
    FrameBuffer& fb,
    const Camera& camera,
    const LightParams& light,
    float time_of_day,
    cudaStream_t stream = 0
);

#endif