#ifndef DAY_NIGHT_CYCLE_CUH
#define DAY_NIGHT_CYCLE_CUH

#include <cuda_runtime.h>
#include "render_common.cuh"

struct DayNightState {
    float time_of_day;
    float day_length;
    float sun_angle;
    float dawn_start;
    float dawn_end;
    float dusk_start;
    float dusk_end;
};

void updateDayNightCycle(
    DayNightState& state,
    LightParams& light,
    float dt
);

void launchApplyDayNightLighting(
    FrameBuffer& fb,
    const LightParams& light,
    const DayNightState& state,
    cudaStream_t stream = 0
);

void launchStarField(
    FrameBuffer& fb,
    const Camera& camera,
    float night_factor,
    unsigned int seed,
    cudaStream_t stream = 0
);

#endif