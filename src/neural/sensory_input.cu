#include "sensory_input.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void sensoryInputKernel(
    float* __restrict__ output,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ energy,
    const float* __restrict__ health,
    const int* __restrict__ age,
    const int* __restrict__ alive,
    const float* __restrict__ genomes,
    const float* __restrict__ repro_cooldown,
    const float* __restrict__ heightmap,
    const float* __restrict__ vegetation,
    const float* __restrict__ temperature,
    int world_size,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    float* out = output + idx * NEURAL_INPUT_SIZE;

    float px = pos_x[idx];
    float py = pos_y[idx];
    int gx = min(max((int)px, 0), world_size - 1);
    int gy = min(max((int)py, 0), world_size - 1);
    int cell = gy * world_size + gx;

    out[0] = px / (float)world_size;
    out[1] = py / (float)world_size;

    float h = heightmap ? heightmap[cell] : 0.5f;
    out[2] = h;

    float h_r = (gx + 1 < world_size && heightmap) ? heightmap[gy * world_size + gx + 1] : h;
    float h_u = (gy + 1 < world_size && heightmap) ? heightmap[(gy + 1) * world_size + gx] : h;
    out[3] = h_r - h;
    out[4] = h_u - h;
    out[5] = sqrtf(out[3] * out[3] + out[4] * out[4]);

    out[6] = vegetation ? vegetation[cell] : 0.0f;
    out[7] = (vegetation && gx + 1 < world_size) ? vegetation[gy * world_size + gx + 1] : 0.0f;
    out[8] = (vegetation && gy + 1 < world_size) ? vegetation[(gy + 1) * world_size + gx] : 0.0f;
    out[9] = (vegetation && gx > 0) ? vegetation[gy * world_size + gx - 1] : 0.0f;

    out[10] = temperature ? temperature[cell] / 50.0f : 0.4f;
    out[11] = heightmap ? ((h < WATER_LEVEL) ? 1.0f : 0.0f) : 0.0f;

    out[12] = 0.0f;
    out[13] = 0.0f;
    out[14] = 0.0f;
    out[15] = 0.0f;
    out[16] = 0.0f;
    out[17] = 0.0f;
    out[18] = 0.0f;
    out[19] = 0.0f;

    out[20] = energy[idx] / MAX_ENERGY;
    out[21] = health[idx];

    const float* genome = genomes + idx * GENOME_SIZE;
    float max_lifespan = 500.0f + genome[GENE_LIFESPAN] * 2000.0f;
    out[22] = (float)age[idx] / max_lifespan;

    float cd = repro_cooldown[idx];
    out[23] = (energy[idx] > REPRODUCE_ENERGY_COST && cd <= 0.0f) ? 1.0f : 0.0f;
}

void launchSensoryInputKernel(
    float* d_sensory_output,
    const SensoryContext& ctx,
    cudaStream_t stream
) {
    if (ctx.num_creatures <= 0) return;
    int block = 256;
    int grid = (ctx.num_creatures + block - 1) / block;

    const CreatureData& c = *ctx.creatures;

    sensoryInputKernel<<<grid, block, 0, stream>>>(
        d_sensory_output,
        c.d_pos_x, c.d_pos_y,
        c.d_energy, c.d_health,
        c.d_age, c.d_alive, c.d_genomes,
        c.d_repro_cooldown,
        ctx.world.d_heightmap,
        ctx.world.d_vegetation,
        ctx.world.d_temperature,
        ctx.world.world_size,
        ctx.num_creatures
    );
}