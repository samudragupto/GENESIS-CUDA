#include "creature_movement.cuh"
#include "../core/constants.cuh"
#include "../core/cuda_utils.cuh"

__device__ float sampleHeightAt(const float* __restrict__ heightmap, float x, float y, int world_size) {
    int gx = min(max((int)x, 0), world_size - 1);
    int gy = min(max((int)y, 0), world_size - 1);
    return heightmap[gy * world_size + gx];
}

__device__ float2 computeTerrainGradient(const float* __restrict__ heightmap, float x, float y, int world_size) {
    float h_c = sampleHeightAt(heightmap, x, y, world_size);
    float h_r = sampleHeightAt(heightmap, x + 1.0f, y, world_size);
    float h_u = sampleHeightAt(heightmap, x, y + 1.0f, world_size);
    return make_float2(h_r - h_c, h_u - h_c);
}

__global__ void movementKernel(
    float* __restrict__ pos_x,
    float* __restrict__ pos_y,
    float* __restrict__ vel_x,
    float* __restrict__ vel_y,
    float* __restrict__ energy,
    const int* __restrict__ alive,
    const int* __restrict__ state,
    const float* __restrict__ genomes,
    const float* __restrict__ action_move_dx,
    const float* __restrict__ action_move_dy,
    const float* __restrict__ action_speed,
    const float* __restrict__ heightmap,
    int world_size,
    float dt,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (state[idx] == STATE_DEAD) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float max_speed = 1.0f + genome[GENE_SPEED] * 4.0f;
    float body_size = 0.5f + genome[GENE_SIZE] * 2.0f;
    float move_cost_factor = body_size * 0.001f;

    float desired_dx = action_move_dx[idx];
    float desired_dy = action_move_dy[idx];
    float desired_speed = fmaxf(action_speed[idx], 0.0f) * max_speed;

    float mag = sqrtf(desired_dx * desired_dx + desired_dy * desired_dy);
    if (mag > 1e-6f) {
        desired_dx /= mag;
        desired_dy /= mag;
    } else {
        desired_dx = 0.0f;
        desired_dy = 0.0f;
        desired_speed = 0.0f;
    }

    float target_vx = desired_dx * desired_speed;
    float target_vy = desired_dy * desired_speed;

    float acceleration = 5.0f / body_size;
    float cur_vx = vel_x[idx];
    float cur_vy = vel_y[idx];

    float new_vx = cur_vx + (target_vx - cur_vx) * fminf(acceleration * dt, 1.0f);
    float new_vy = cur_vy + (target_vy - cur_vy) * fminf(acceleration * dt, 1.0f);

    float2 grad = computeTerrainGradient(heightmap, pos_x[idx], pos_y[idx], world_size);
    float slope_factor = 1.0f - fminf(sqrtf(grad.x * grad.x + grad.y * grad.y) * 3.0f, 0.8f);

    new_vx *= slope_factor;
    new_vy *= slope_factor;

    new_vx -= grad.x * 2.0f;
    new_vy -= grad.y * 2.0f;

    float speed = sqrtf(new_vx * new_vx + new_vy * new_vy);
    if (speed > max_speed) {
        float scale = max_speed / speed;
        new_vx *= scale;
        new_vy *= scale;
        speed = max_speed;
    }

    float friction = 0.98f;
    new_vx *= friction;
    new_vy *= friction;

    float new_px = pos_x[idx] + new_vx * dt;
    float new_py = pos_y[idx] + new_vy * dt;

    float border = 2.0f;
    new_px = fminf(fmaxf(new_px, border), (float)(world_size - 1) - border);
    new_py = fminf(fmaxf(new_py, border), (float)(world_size - 1) - border);

    float h = sampleHeightAt(heightmap, new_px, new_py, world_size);
    if (h < WATER_LEVEL) {
        float swim_gene = genome[GENE_LIMBS];
        if (swim_gene < 0.3f) {
            new_px = pos_x[idx];
            new_py = pos_y[idx];
            new_vx = -new_vx * 0.5f;
            new_vy = -new_vy * 0.5f;
        } else {
            new_vx *= 0.5f;
            new_vy *= 0.5f;
            move_cost_factor *= 2.0f;
        }
    }

    pos_x[idx] = new_px;
    pos_y[idx] = new_py;
    vel_x[idx] = new_vx;
    vel_y[idx] = new_vy;

    float energy_cost = speed * move_cost_factor * dt;
    energy[idx] -= energy_cost;
}

void launchMovementKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    const float* d_heightmap,
    int world_size,
    float dt,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    movementKernel<<<grid, block, 0, stream>>>(
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_vel_x, creatures.d_vel_y,
        creatures.d_energy, creatures.d_alive,
        creatures.d_state, creatures.d_genomes,
        actions.d_move_dx, actions.d_move_dy,
        actions.d_speed, d_heightmap,
        world_size, dt, num_creatures
    );
}