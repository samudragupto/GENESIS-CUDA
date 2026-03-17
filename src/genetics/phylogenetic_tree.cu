#include "phylogenetic_tree.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void initPhyloTree(PhyloTreeGPU& tree, int max_nodes) {
    tree.max_nodes = max_nodes;
    tree.num_nodes = 0;
    CUDA_CHECK(cudaMalloc(&tree.d_nodes, max_nodes * sizeof(PhyloNode)));
    CUDA_CHECK(cudaMalloc(&tree.d_num_nodes, sizeof(int)));
    CUDA_CHECK(cudaMemset(tree.d_nodes, 0, max_nodes * sizeof(PhyloNode)));
    CUDA_CHECK(cudaMemset(tree.d_num_nodes, 0, sizeof(int)));
}

void freePhyloTree(PhyloTreeGPU& tree) {
    cudaFree(tree.d_nodes);
    cudaFree(tree.d_num_nodes);
    tree.d_nodes = nullptr;
    tree.d_num_nodes = nullptr;
    tree.num_nodes = 0;
}

__global__ void recordSpeciationKernel(
    PhyloNode* __restrict__ nodes,
    int* __restrict__ num_nodes,
    int max_nodes,
    int parent_species,
    int child_species,
    int tick,
    const float* __restrict__ genomes,
    int genome_index
) {
    if (threadIdx.x != 0) return;

    int idx = atomicAdd(num_nodes, 1);
    if (idx >= max_nodes) return;

    nodes[idx].species_id = child_species;
    nodes[idx].parent_species = parent_species;
    nodes[idx].birth_tick = tick;
    nodes[idx].death_tick = -1;
    nodes[idx].active = 1;
    nodes[idx].population = 1;

    if (genomes) {
        int base = genome_index * GENOME_LENGTH;
        for (int g = 0; g < CENTROID_GENES; g++) {
            nodes[idx].centroid[g] = genomes[base + g];
        }
    } else {
        for (int g = 0; g < CENTROID_GENES; g++) {
            nodes[idx].centroid[g] = 0.0f;
        }
    }
}

__global__ void recordExtinctionKernel(
    PhyloNode* __restrict__ nodes,
    int num_nodes,
    const int* __restrict__ species_pop,
    int max_species,
    int tick
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_nodes) return;
    if (!nodes[idx].active) return;

    int sid = nodes[idx].species_id;
    if (sid < 0 || sid >= max_species) return;

    if (species_pop[sid] <= 0) {
        nodes[idx].active = 0;
        nodes[idx].death_tick = tick;
    } else {
        nodes[idx].population = species_pop[sid];
    }
}

void launchRecordSpeciation(
    PhyloTreeGPU& tree,
    int parent_species,
    int child_species,
    int tick,
    const float* d_genomes,
    int genome_index,
    cudaStream_t stream
) {
    recordSpeciationKernel<<<1, 1, 0, stream>>>(
        tree.d_nodes, tree.d_num_nodes, tree.max_nodes,
        parent_species, child_species, tick,
        d_genomes, genome_index
    );

    int h_num;
    CUDA_CHECK(cudaMemcpyAsync(&h_num, tree.d_num_nodes, sizeof(int),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tree.num_nodes = h_num;
}

void launchRecordExtinction(
    PhyloTreeGPU& tree,
    const int* d_species_pop_count,
    int max_species,
    int tick,
    cudaStream_t stream
) {
    if (tree.num_nodes <= 0) return;
    int block = 256;
    int grid = (tree.num_nodes + block - 1) / block;
    recordExtinctionKernel<<<grid, block, 0, stream>>>(
        tree.d_nodes, tree.num_nodes,
        d_species_pop_count, max_species, tick
    );
}