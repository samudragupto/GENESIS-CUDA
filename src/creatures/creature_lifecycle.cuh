#ifndef CREATURE_LIFECYCLE_CUH
#define CREATURE_LIFECYCLE_CUH

#include <cuda_runtime.h>
#include "creature_common.cuh"

struct LifecycleParams {
    float base_energy_drain;
    float age_energy_factor;
    float health_regen_rate;
    float starvation_damage;
    float dt;
};

void launchLifecycleKernel(
    CreatureData& creatures,
    const LifecycleParams& params,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchDeathKernel(
    CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream = 0
);

void launchCooldownKernel(
    CreatureData& creatures,
    float dt,
    int num_creatures,
    cudaStream_t stream = 0
);

#endif