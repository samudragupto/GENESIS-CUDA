#include "creature_lifecycle.cuh"
#include "../core/constants.cuh"
#include "../core/cuda_utils.cuh"

__global__ void lifecycleKernel(
    float* __restrict__ energy,
    float* __restrict__ health,
    int* __restrict__ age,
    int* __restrict__ state,
    const int* __restrict__ alive,
    const float* __restrict__ genomes,
    float base_energy_drain,
    float age_energy_factor,
    float health_regen_rate,
    float starvation_damage,
    float dt,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (state[idx] == STATE_DEAD) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float body_size = 0.5f + genome[GENE_SIZE] * 2.0f;
    float efficiency = genome[GENE_EFFICIENCY];
    float max_lifespan_gene = genome[GENE_LIFESPAN];
    int max_age = (int)(500.0f + max_lifespan_gene * 2000.0f);

    age[idx] += 1;

    float age_fraction = (float)age[idx] / (float)max_age;
    float metabolic_cost = base_energy_drain * body_size * (1.0f - efficiency * 0.5f);
    metabolic_cost *= (1.0f + age_fraction * age_energy_factor);

    energy[idx] -= metabolic_cost * dt;

    if (energy[idx] < 0.0f) {
        health[idx] -= starvation_damage * dt;
        energy[idx] = 0.0f;
    }

    if (energy[idx] > 0.2f && health[idx] < 1.0f) {
        float regen = health_regen_rate * dt * efficiency;
        health[idx] = fminf(health[idx] + regen, 1.0f);
    }

    if (age[idx] > max_age) {
        float death_prob = (float)(age[idx] - max_age) / (float)(max_age * 0.2f);
        if (death_prob > 0.95f) {
            health[idx] = 0.0f;
        } else {
            health[idx] -= 0.01f * dt;
        }
    }

    if (state[idx] != STATE_COMBAT && state[idx] != STATE_MATING) {
        state[idx] = STATE_FORAGING;
    }
}

__global__ void deathKernel(
    int* __restrict__ alive,
    int* __restrict__ state,
    const float* __restrict__ health,
    const float* __restrict__ energy,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    if (health[idx] <= 0.0f) {
        alive[idx] = 0;
        state[idx] = STATE_DEAD;
    }
}

__global__ void cooldownKernel(
    float* __restrict__ repro_cooldown,
    const int* __restrict__ alive,
    float dt,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    if (repro_cooldown[idx] > 0.0f) {
        repro_cooldown[idx] -= dt;
        if (repro_cooldown[idx] < 0.0f) repro_cooldown[idx] = 0.0f;
    }
}

void launchLifecycleKernel(
    CreatureData& creatures,
    const LifecycleParams& params,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    lifecycleKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_health,
        creatures.d_age, creatures.d_state,
        creatures.d_alive, creatures.d_genomes,
        params.base_energy_drain, params.age_energy_factor,
        params.health_regen_rate, params.starvation_damage,
        params.dt, num_creatures
    );
}

void launchDeathKernel(
    CreatureData& creatures,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    deathKernel<<<grid, block, 0, stream>>>(
        creatures.d_alive, creatures.d_state,
        creatures.d_health, creatures.d_energy,
        num_creatures
    );
}

void launchCooldownKernel(
    CreatureData& creatures,
    float dt,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    cooldownKernel<<<grid, block, 0, stream>>>(
        creatures.d_repro_cooldown, creatures.d_alive,
        dt, num_creatures
    );
}