#include "disease_model.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void allocateDiseaseData(DiseaseData& data, int max_creatures) {
    data.max_creatures = max_creatures;
    CUDA_CHECK(cudaMalloc(&data.d_disease_state, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&data.d_infection_timer, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data.d_immunity, max_creatures * sizeof(float)));

    CUDA_CHECK(cudaMemset(data.d_disease_state, 0, max_creatures * sizeof(int)));
    CUDA_CHECK(cudaMemset(data.d_infection_timer, 0, max_creatures * sizeof(float)));
    CUDA_CHECK(cudaMemset(data.d_immunity, 0, max_creatures * sizeof(float)));
}

void freeDiseaseData(DiseaseData& data) {
    cudaFree(data.d_disease_state);
    cudaFree(data.d_infection_timer);
    cudaFree(data.d_immunity);
}

__global__ void diseaseSeedKernel(
    int* __restrict__ disease_state,
    float* __restrict__ infection_timer,
    const int* __restrict__ alive,
    int num_creatures,
    int num_to_infect,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    curandState local_rng = rng[idx];
    float r = curand_uniform(&local_rng);

    float prob = (float)num_to_infect / (float)num_creatures;
    if (r < prob) {
        disease_state[idx] = DISEASE_INFECTED;
        infection_timer[idx] = 0.0f;
    }

    rng[idx] = local_rng;
}

__global__ void diseaseSpreadKernel(
    int* __restrict__ disease_state,
    float* __restrict__ infection_timer,
    const float* __restrict__ immunity,
    const float* __restrict__ pos_x,
    const float* __restrict__ pos_y,
    const int* __restrict__ alive,
    const float* __restrict__ genomes,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ sorted_indices,
    int grid_size,
    float cell_size,
    float transmission_radius,
    float transmission_prob,
    int num_creatures,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;
    if (disease_state[idx] != DISEASE_SUSCEPTIBLE) return;

    float my_x = pos_x[idx];
    float my_y = pos_y[idx];
    float my_immunity = immunity[idx];

    if (my_immunity > 0.5f) return;

    int cx = (int)(my_x / cell_size);
    int cy = (int)(my_y / cell_size);

    curandState local_rng = rng[idx];
    float trans_radius_sq = transmission_radius * transmission_radius;

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
                if (disease_state[other] != DISEASE_INFECTED) continue;

                float ox = pos_x[other] - my_x;
                float oy = pos_y[other] - my_y;
                float dist2 = ox * ox + oy * oy;
                if (dist2 > trans_radius_sq) continue;

                float distance_factor = 1.0f - sqrtf(dist2) / transmission_radius;
                float social_gene = genomes[idx * GENOME_SIZE + GENE_SOCIAL];
                float effective_prob = transmission_prob * distance_factor * (0.5f + social_gene * 0.5f);
                effective_prob *= (1.0f - my_immunity);

                if (curand_uniform(&local_rng) < effective_prob) {
                    disease_state[idx] = DISEASE_INFECTED;
                    infection_timer[idx] = 0.0f;
                    rng[idx] = local_rng;
                    return;
                }
            }
        }
    }

    rng[idx] = local_rng;
}

__global__ void diseaseProgressKernel(
    int* __restrict__ disease_state,
    float* __restrict__ infection_timer,
    float* __restrict__ immunity,
    float* __restrict__ health,
    float* __restrict__ energy,
    const int* __restrict__ alive,
    const float* __restrict__ genomes,
    float infection_duration,
    float mortality_rate,
    float health_damage_rate,
    float energy_drain_rate,
    float immunity_duration,
    float dt,
    int num_creatures,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) return;

    int state = disease_state[idx];

    if (state == DISEASE_INFECTED) {
        infection_timer[idx] += dt;

        const float* genome = genomes + idx * GENOME_SIZE;
        float resistance = genome[GENE_ARMOR] * 0.5f;

        health[idx] -= health_damage_rate * (1.0f - resistance) * dt;
        energy[idx] -= energy_drain_rate * dt;

        if (infection_timer[idx] > infection_duration) {
            curandState local_rng = rng[idx];
            float death_chance = mortality_rate * (1.0f - resistance);

            if (curand_uniform(&local_rng) < death_chance) {
                disease_state[idx] = DISEASE_DEAD;
                health[idx] = 0.0f;
            } else {
                disease_state[idx] = DISEASE_RECOVERED;
                immunity[idx] = 1.0f;
            }
            rng[idx] = local_rng;
        }
    } else if (state == DISEASE_RECOVERED) {
        immunity[idx] -= (1.0f / immunity_duration) * dt;
        if (immunity[idx] <= 0.0f) {
            immunity[idx] = 0.0f;
            disease_state[idx] = DISEASE_SUSCEPTIBLE;
        }
    }
}

void launchDiseaseSeed(
    DiseaseData& disease,
    int num_creatures,
    int num_to_infect,
    curandState* d_rng,
    cudaStream_t stream
) {
    if (num_creatures <= 0 || num_to_infect <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    diseaseSeedKernel<<<grid, block, 0, stream>>>(
        disease.d_disease_state, disease.d_infection_timer,
        nullptr, num_creatures, num_to_infect, d_rng
    );
}

void launchDiseaseSpread(
    DiseaseData& disease,
    const CreatureData& creatures,
    const int* d_cell_start,
    const int* d_cell_end,
    const int* d_sorted_indices,
    int grid_size,
    float cell_size,
    const DiseaseParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid_dim = (num_creatures + block - 1) / block;
    diseaseSpreadKernel<<<grid_dim, block, 0, stream>>>(
        disease.d_disease_state, disease.d_infection_timer,
        disease.d_immunity, creatures.d_pos_x, creatures.d_pos_y,
        creatures.d_alive, creatures.d_genomes,
        d_cell_start, d_cell_end, d_sorted_indices,
        grid_size, cell_size,
        params.transmission_radius, params.transmission_prob,
        num_creatures, d_rng
    );
}

void launchDiseaseProgress(
    DiseaseData& disease,
    CreatureData& creatures,
    const DiseaseParams& params,
    int num_creatures,
    curandState* d_rng,
    cudaStream_t stream
) {
    if (num_creatures <= 0) return;
    int block = 256;
    int grid = (num_creatures + block - 1) / block;
    diseaseProgressKernel<<<grid, block, 0, stream>>>(
        disease.d_disease_state, disease.d_infection_timer,
        disease.d_immunity, creatures.d_health, creatures.d_energy,
        creatures.d_alive, creatures.d_genomes,
        params.infection_duration, params.mortality_rate,
        params.health_damage_rate, params.energy_drain_rate,
        params.immunity_duration, params.dt,
        num_creatures, d_rng
    );
}