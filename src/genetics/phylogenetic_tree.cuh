#ifndef PHYLOGENETIC_TREE_CUH
#define PHYLOGENETIC_TREE_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct PhyloNode {
    int species_id;
    int parent_species;
    int birth_tick;
    int death_tick;
    int active;
    int population;
    float centroid[CENTROID_GENES];
};

struct PhyloTreeGPU {
    PhyloNode* d_nodes;
    int* d_num_nodes;
    int max_nodes;
    int num_nodes;
};

void initPhyloTree(PhyloTreeGPU& tree, int max_nodes);
void freePhyloTree(PhyloTreeGPU& tree);

void launchRecordSpeciation(
    PhyloTreeGPU& tree,
    int parent_species,
    int child_species,
    int tick,
    const float* d_genomes,
    int genome_index,
    cudaStream_t stream = 0
);

void launchRecordExtinction(
    PhyloTreeGPU& tree,
    const int* d_species_pop_count,
    int max_species,
    int tick,
    cudaStream_t stream = 0
);

#endif