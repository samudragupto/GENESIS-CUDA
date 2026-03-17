#ifndef PHYLO_VISUALIZER_CUH
#define PHYLO_VISUALIZER_CUH

#include <cuda_runtime.h>
#include "../genetics/phylogenetic_tree.cuh"

struct PhyloRenderNode {
    float x, y;
    float vx, vy;
    float mass;
    int parent;
    int species_id;
    int is_alive;
    float r, g, b;
};

struct PhyloVisualizerData {
    PhyloRenderNode* d_nodes;
    float* d_edge_buffer;
    int* d_num_edges;
    int max_nodes;
    int max_edges;
};

void allocatePhyloVisualizerData(PhyloVisualizerData& pv, int max_nodes);
void freePhyloVisualizerData(PhyloVisualizerData& pv);

void launchBuildPhyloRenderNodes(
    PhyloVisualizerData& pv,
    const PhyloTreeGPU& tree,
    const int* d_species_pop_count,
    int max_species,
    cudaStream_t stream = 0
);

void launchForceDirectedLayout(
    PhyloVisualizerData& pv,
    int num_nodes,
    int iterations,
    float repulsion_strength,
    float attraction_strength,
    float damping,
    cudaStream_t stream = 0
);

void launchBuildEdgeBuffer(
    PhyloVisualizerData& pv,
    int num_nodes,
    cudaStream_t stream = 0
);

#endif