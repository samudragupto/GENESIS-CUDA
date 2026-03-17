#include "fitness_evaluation.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

__global__ void fitnessEvalKernel(
    const float* __restrict__ genomes,
    const float* __restrict__ energy,
    const int* __restrict__ age,
    const int* __restrict__ alive,
    float* __restrict__ fitness,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;
    if (!alive[idx]) { fitness[idx] = 0.0f; return; }

    const float* g = genomes + idx * GENOME_LENGTH;
    float max_life = 500.0f + g[GENE_LIFESPAN] * 2000.0f;
    float age_frac = (float)age[idx] / max_life;
    float efficiency = g[GENE_EFFICIENCY];
    float e = energy[idx] / MAX_ENERGY;

    fitness[idx] = e * 0.4f + efficiency * 0.3f + g[GENE_SPEED] * 0.15f +
                   g[GENE_SENSE] * 0.15f - age_frac * 0.1f;
    fitness[idx] = fmaxf(fitness[idx], 0.0f);
}

__global__ void tournamentKernel(
    const float* __restrict__ fitness,
    const int* __restrict__ alive,
    int* __restrict__ selected,
    int num_creatures,
    int num_to_select,
    int tournament_size,
    curandState* __restrict__ rng
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_to_select) return;

    curandState local_rng = rng[idx];
    int best = -1;
    float best_fit = -1.0f;

    for (int t = 0; t < tournament_size; t++) {
        int candidate = (int)(curand_uniform(&local_rng) * (num_creatures - 1));
        if (alive[candidate] && fitness[candidate] > best_fit) {
            best_fit = fitness[candidate];
            best = candidate;
        }
    }
    selected[idx] = (best >= 0) ? best : 0;
    rng[idx] = local_rng;
}

__global__ void eligibilityKernel(
    const float* __restrict__ energy,
    const int* __restrict__ age,
    const int* __restrict__ alive,
    const float* __restrict__ genomes,
    int* __restrict__ eligible,
    int* __restrict__ eligible_count,
    int num_creatures
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_creatures) return;

    eligible[idx] = 0;
    if (!alive[idx]) return;

    const float* g = genomes + idx * GENOME_LENGTH;
    float repro_cost = g[GENE_REPRO_COST] * 0.5f + 0.1f;
    float maturity = g[GENE_MATURITY] * 200.0f + 50.0f;

    if (energy[idx] > repro_cost && (float)age[idx] > maturity) {
        eligible[idx] = 1;
        atomicAdd(eligible_count, 1);
    }
}

void launchFitnessEvaluation(const float* d_genomes, const float* d_energy,
    const int* d_age, const int* d_alive, float* d_fitness, int n, cudaStream_t stream) {
    if (n <= 0) return;
    fitnessEvalKernel<<<(n+255)/256, 256, 0, stream>>>(d_genomes, d_energy, d_age, d_alive, d_fitness, n);
}

void launchTournamentSelection(const float* d_fitness, const int* d_alive, int* d_selected,
    int n, int num_sel, int tourn, curandState* rng, cudaStream_t stream) {
    if (num_sel <= 0) return;
    tournamentKernel<<<(num_sel+255)/256, 256, 0, stream>>>(d_fitness, d_alive, d_selected, n, num_sel, tourn, rng);
}

void launchEligibilityCheck(const float* d_energy, const int* d_age, const int* d_alive,
    const float* d_genomes, int* d_eligible, int* d_eligible_count, int n, cudaStream_t stream) {
    if (n <= 0) return;
    CUDA_CHECK(cudaMemsetAsync(d_eligible_count, 0, sizeof(int), stream));
    eligibilityKernel<<<(n+255)/256, 256, 0, stream>>>(d_energy, d_age, d_alive, d_genomes, d_eligible, d_eligible_count, n);
}