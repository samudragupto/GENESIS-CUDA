#include "creature_interaction.cuh"
#include "../core/constants.cuh"
#include "../core/cuda_utils.cuh"

__global__ void feedingKernel(
    float* __restrict__ energy,
    const int* __restrict__ alive,
    int* __restrict__ state,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ genomes,
    const float* __restrict__ action_eat,
    float* __restrict__ vegetation,
    int world_size,
    float eat_energy_gain,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (state[idx] == STATE_DEAD) return;

    if (action_eat[idx] < 0.5f) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float diet = genome[GENE_DIET];

    if (diet > 0.5f) return;

    int gx = min(max((int)pos_x[idx], 0), world_size - 1);
    int gy = min(max((int)pos_y[idx], 0), world_size - 1);
    int cell = gy * world_size + gx;

    float available = vegetation[cell];
    if (available < 0.01f) return;

    float efficiency = genome[GENE_EFFICIENCY];
    float eat_amount = fminf(available, 0.1f * (0.5f + efficiency));

    float old_val = atomicAdd(&vegetation[cell], -eat_amount);
    if (old_val < eat_amount) {
        atomicAdd(&vegetation[cell], eat_amount - old_val);
        eat_amount = fmaxf(old_val, 0.0f);
    }

    float gained = eat_amount * eat_energy_gain * (0.5f + efficiency);
    energy[idx] += gained;
    energy[idx] = fminf(energy[idx], MAX_ENERGY);

    state[idx] = STATE_EATING;
}

__global__ void combatKernel(
    float* __restrict__ energy,
    float* __restrict__ health,
    const int* __restrict__ alive,
    int* __restrict__ state,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ genomes,
    const float* __restrict__ action_attack,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_indices,
    int grid_size,
    float cell_size,
    float attack_radius,
    float attack_damage,
    float attack_energy_cost,
    int num_creatures,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (state[idx] == STATE_DEAD) return;
    if (action_attack[idx] < 0.5f) return;

    const float* genome = genomes + idx * GENOME_SIZE;
    float diet = genome[GENE_DIET];
    if (diet < 0.3f) return;

    float my_x = pos_x[idx];
    float my_y = pos_y[idx];
    float my_size = 0.5f + genome[GENE_SIZE] * 2.0f;
    float my_armor = genome[GENE_ARMOR];
    float aggression = genome[GENE_AGGRO];

    int cx = (int)(my_x / cell_size);
    int cy = (int)(my_y / cell_size);

    int best_target = -1;
    float best_dist = attack_radius * attack_radius;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;
            int cell_idx = ny * grid_size + nx;
            int start = cell_start[cell_idx];
            int end = cell_end[cell_idx];
            if (start < 0) continue;

            for (int j = start; j < end; j++) {
                int other = sorted_indices[j];
                if (other == idx) continue;
                if (!alive[other]) continue;
                if (state[other] == STATE_DEAD) continue;

                float ox = pos_x[other] - my_x;
                float oy = pos_y[other] - my_y;
                float dist2 = ox * ox + oy * oy;
                if (dist2 < best_dist) {
                    best_dist = dist2;
                    best_target = other;
                }
            }
        }
    }

    if (best_target < 0) return;

    curandState local_rng = rng[idx];

    const float* target_genome = genomes + best_target * GENOME_SIZE;
    float target_size = 0.5f + target_genome[GENE_SIZE] * 2.0f;
    float target_armor = target_genome[GENE_ARMOR];

    float attack_power = my_size * aggression * (1.0f + curand_uniform(&local_rng) * 0.3f);
    float defense = target_size * target_armor * 0.5f;
    float damage_dealt = fmaxf(attack_damage * (attack_power - defense), 0.0f);

    atomicAdd(&health[best_target], -damage_dealt);
    energy[idx] -= attack_energy_cost;

    if (diet > 0.5f && health[best_target] <= 0.0f) {
        float meat_energy = target_size * 0.3f;
        energy[idx] += meat_energy;
        energy[idx] = fminf(energy[idx], MAX_ENERGY);
    }

    state[idx] = STATE_COMBAT;
    rng[idx] = local_rng;
}

__global__ void matingKernel(
    float* __restrict__ energy,
    const int* __restrict__ alive,
    int* __restrict__ state,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const float* __restrict__ genomes,
    const int* __restrict__ species_id,
    const float* __restrict__ repro_cooldown,
    const float* __restrict__ action_reproduce,
    int* __restrict__ reproduce_flag,
    int* __restrict__ mate_index,
    int* __restrict__ mating_lock,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_indices,
    int grid_size,
    float cell_size,
    float mate_radius,
    float mate_energy_cost,
    float min_repro_energy,
    int num_creatures,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (state[idx] == STATE_DEAD) return;
    if (action_reproduce[idx] < 0.5f) return;
    if (energy[idx] < min_repro_energy) return;
    if (repro_cooldown[idx] > 0.0f) return;

    reproduce_flag[idx] = 0;
    mate_index[idx] = -1;

    float my_x = pos_x[idx];
    float my_y = pos_y[idx];
    int my_species = species_id[idx];

    int cx = (int)(my_x / cell_size);
    int cy = (int)(my_y / cell_size);

    curandState local_rng = rng[idx];
    int best_mate = -1;
    float best_dist = mate_radius * mate_radius;

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= grid_size || ny < 0 || ny >= grid_size) continue;
            int cell_idx = ny * grid_size + nx;
            int start = cell_start[cell_idx];
            int end = cell_end[cell_idx];
            if (start < 0) continue;

            for (int j = start; j < end; j++) {
                int other = sorted_indices[j];
                if (other == idx) continue;
                if (!alive[other]) continue;
                if (state[other] == STATE_DEAD) continue;
                if (energy[other] < min_repro_energy * 0.5f) continue;

                int other_species = species_id[other];
                float genetic_compat = 1.0f;
                if (my_species != other_species) {
                    float dist_genes = 0.0f;
                    const float* g1 = genomes + idx * GENOME_SIZE;
                    const float* g2 = genomes + other * GENOME_SIZE;
                    for (int g = 0; g < 32; g++) {
                        float d = g1[g] - g2[g];
                        dist_genes += d * d;
                    }
                    genetic_compat = expf(-dist_genes * 0.5f);
                }

                if (genetic_compat < 0.3f) continue;

                float ox = pos_x[other] - my_x;
                float oy = pos_y[other] - my_y;
                float dist2 = ox * ox + oy * oy;
                if (dist2 < best_dist) {
                    best_dist = dist2;
                    best_mate = other;
                }
            }
        }
    }

    if (best_mate < 0) {
        rng[idx] = local_rng;
        return;
    }

    int lock_a = min(idx, best_mate);
    int lock_b = max(idx, best_mate);

    int old_a = atomicCAS(&mating_lock[lock_a], 0, 1);
    if (old_a != 0) {
        rng[idx] = local_rng;
        return;
    }
    int old_b = atomicCAS(&mating_lock[lock_b], 0, 1);
    if (old_b != 0) {
        atomicExch(&mating_lock[lock_a], 0);
        rng[idx] = local_rng;
        return;
    }

    reproduce_flag[idx] = 1;
    mate_index[idx] = best_mate;
    energy[idx] -= mate_energy_cost;
    energy[best_mate] -= mate_energy_cost * 0.5f;
    state[idx] = STATE_MATING;

    rng[idx] = local_rng;
}

void launchFeedingKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    float* d_vegetation,
    int world_size,
    const InteractionParams& params,
    int num_creatures,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    feedingKernel<<<grid, block, 0, stream>>>(
        creatures.d_energy, creatures.d_alive,
        creatures.d_state, creatures.d_pos_x,
        creatures.d_pos_y, creatures.d_genomes,
        actions.d_eat, d_vegetation,
        world_size, params.eat_energy_gain, num_creatures
    );
}

void launchCombatKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const InteractionParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid_dim = (num_creatures + block - 1) / block;
    combatKernel<<<grid_dim, block, 0, stream>>>(
        creatures.d_energy, creatures.d_health,
        creatures.d_alive, creatures.d_state,
        creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_genomes, actions.d_attack,
        d_cell_start, d_cell_end, d_sorted_indices,
        grid_size, cell_size,
        params.attack_radius, params.attack_damage,
        params.attack_energy_cost, num_creatures, d_rng
    );
}

void launchMatingKernel(
    CreatureData& creatures,
    const CreatureActions& actions,
    InteractionBuffers& interaction,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const InteractionParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(interaction.d_reproduce_flag, 0, num_creatures * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(interaction.d_mate_index, 0xFF, num_creatures * sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(interaction.d_mating_lock, 0, num_creatures * sizeof(int), stream));

    int block = 256;
    int grid_dim = (num_creatures + block - 1) / block;
    float min_repro_energy = 0.4f;

    matingKernel<<<grid_dim, block, 0, stream>>>(
        creatures.d_energy, creatures.d_alive,
        creatures.d_state, creatures.d_pos_x,
        creatures.d_pos_y, creatures.d_genomes,
        creatures.d_species_id, creatures.d_repro_cooldown,
        actions.d_reproduce,
        interaction.d_reproduce_flag, interaction.d_mate_index,
        interaction.d_mating_lock,
        d_cell_start, d_cell_end, d_sorted_indices,
        grid_size, cell_size,
        params.mate_radius, params.mate_energy_cost,
        min_repro_energy, num_creatures, d_rng
    );
}