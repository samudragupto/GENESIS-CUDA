#include "phylo_visualizer.cuh"
#include "../core/cuda_utils.cuh"

void allocatePhyloVisualizerData(PhyloVisualizerData& pv, int max_nodes) {
    pv.max_nodes = max_nodes;
    pv.max_edges = max_nodes * 2;
    CUDA_CHECK(cudaMalloc(&pv.d_nodes, max_nodes * sizeof(PhyloRenderNode)));
    CUDA_CHECK(cudaMalloc(&pv.d_edge_buffer, pv.max_edges * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pv.d_num_edges, sizeof(int)));
    CUDA_CHECK(cudaMemset(pv.d_nodes, 0, max_nodes * sizeof(PhyloRenderNode)));
    CUDA_CHECK(cudaMemset(pv.d_num_edges, 0, sizeof(int)));
}

void freePhyloVisualizerData(PhyloVisualizerData& pv) {
    cudaFree(pv.d_nodes);
    cudaFree(pv.d_edge_buffer);
    cudaFree(pv.d_num_edges);
}

__global__ void buildRenderNodesKernel(
    PhyloRenderNode* __restrict__ nodes,
    const PhyloNode* __restrict__ tree_nodes,
    const int* __restrict__ species_pop,
    int max_species,
    int num_tree_nodes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tree_nodes) return;

    PhyloNode tn = tree_nodes[idx];
    if (!tn.active) return;

    nodes[idx].species_id = tn.species_id;
    nodes[idx].parent = tn.parent_species;

    float angle = (float)idx * 2.3999632f;
    float radius = 10.0f + (float)idx * 0.1f;
    nodes[idx].x = cosf(angle) * radius;
    nodes[idx].y = sinf(angle) * radius;
    nodes[idx].vx = 0.0f;
    nodes[idx].vy = 0.0f;

    int sid = tn.species_id;
    int pop = 0;
    if (sid >= 0 && sid < max_species) {
        pop = species_pop[sid];
    }
    nodes[idx].is_alive = (pop > 0) ? 1 : 0;
    nodes[idx].mass = 1.0f + logf(1.0f + (float)pop) * 2.0f;

    float hash = sinf((float)sid * 12.9898f) * 43758.5453f;
    hash = hash - floorf(hash);
    nodes[idx].r = 0.3f + hash * 0.7f;

    hash = sinf((float)sid * 78.233f) * 43758.5453f;
    hash = hash - floorf(hash);
    nodes[idx].g = 0.3f + hash * 0.7f;

    hash = sinf((float)sid * 45.164f) * 43758.5453f;
    hash = hash - floorf(hash);
    nodes[idx].b = 0.3f + hash * 0.7f;

    if (!nodes[idx].is_alive) {
        nodes[idx].r *= 0.3f;
        nodes[idx].g *= 0.3f;
        nodes[idx].b *= 0.3f;
    }
}

__global__ void forceDirectedKernel(
    PhyloRenderNode* __restrict__ nodes,
    int num_nodes,
    float repulsion,
    float attraction,
    float damping
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    if (nodes[i].species_id < 0) return;

    float fx = 0.0f;
    float fy = 0.0f;

    for (int j = 0; j < num_nodes; j++) {
        if (j == i) continue;
        if (nodes[j].species_id < 0) continue;

        float dx = nodes[i].x - nodes[j].x;
        float dy = nodes[i].y - nodes[j].y;
        float dist2 = dx * dx + dy * dy + 0.01f;
        float dist = sqrtf(dist2);

        float rep_force = repulsion * nodes[i].mass * nodes[j].mass / dist2;
        fx += rep_force * dx / dist;
        fy += rep_force * dy / dist;
    }

    int parent = nodes[i].parent;
    if (parent >= 0 && parent < num_nodes && nodes[parent].species_id >= 0) {
        float dx = nodes[parent].x - nodes[i].x;
        float dy = nodes[parent].y - nodes[i].y;
        float dist = sqrtf(dx * dx + dy * dy + 0.01f);

        float ideal_dist = 5.0f;
        float spring_force = attraction * (dist - ideal_dist);
        fx += spring_force * dx / dist;
        fy += spring_force * dy / dist;
    }

    float center_force = 0.001f;
    fx -= nodes[i].x * center_force;
    fy -= nodes[i].y * center_force;

    nodes[i].vx = (nodes[i].vx + fx) * damping;
    nodes[i].vy = (nodes[i].vy + fy) * damping;

    float max_vel = 5.0f;
    float vel = sqrtf(nodes[i].vx * nodes[i].vx + nodes[i].vy * nodes[i].vy);
    if (vel > max_vel) {
        nodes[i].vx *= max_vel / vel;
        nodes[i].vy *= max_vel / vel;
    }

    nodes[i].x += nodes[i].vx;
    nodes[i].y += nodes[i].vy;
}

__global__ void buildEdgeBufferKernel(
    const PhyloRenderNode* __restrict__ nodes,
    float* __restrict__ edge_buffer,
    int* __restrict__ num_edges,
    int num_nodes,
    int max_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_nodes) return;
    if (nodes[i].species_id < 0) return;

    int parent = nodes[i].parent;
    if (parent < 0 || parent >= num_nodes) return;
    if (nodes[parent].species_id < 0) return;

    int edge_idx = atomicAdd(num_edges, 1);
    if (edge_idx >= max_edges) return;

    int base = edge_idx * 4;
    edge_buffer[base + 0] = nodes[i].x;
    edge_buffer[base + 1] = nodes[i].y;
    edge_buffer[base + 2] = nodes[parent].x;
    edge_buffer[base + 3] = nodes[parent].y;
}

void launchBuildPhyloRenderNodes(
    PhyloVisualizerData& pv,
    const PhyloTreeGPU& tree,
    const int* d_species_pop_count,
    int max_species,
    cudaStream_t stream
) {
    int num = tree.num_nodes;
    if (num <= 0) return;

    int block = 256;
    int grid = (num + block - 1) / block;
    buildRenderNodesKernel<<<grid, block, 0, stream>>>(
        pv.d_nodes, tree.d_nodes,
        d_species_pop_count, max_species, num
    );
}

void launchForceDirectedLayout(
    PhyloVisualizerData& pv,
    int num_nodes,
    int iterations,
    float repulsion_strength,
    float attraction_strength,
    float damping,
    cudaStream_t stream
) {
    if (num_nodes <= 0) return;
    int block = 256;
    int grid = (num_nodes + block - 1) / block;

    for (int iter = 0; iter < iterations; iter++) {
        forceDirectedKernel<<<grid, block, 0, stream>>>(
            pv.d_nodes, num_nodes,
            repulsion_strength, attraction_strength, damping
        );
    }
}

void launchBuildEdgeBuffer(
    PhyloVisualizerData& pv,
    int num_nodes,
    cudaStream_t stream
) {
    if (num_nodes <= 0) return;

    CUDA_CHECK(cudaMemsetAsync(pv.d_num_edges, 0, sizeof(int), stream));

    int block = 256;
    int grid = (num_nodes + block - 1) / block;
    buildEdgeBufferKernel<<<grid, block, 0, stream>>>(
        pv.d_nodes, pv.d_edge_buffer, pv.d_num_edges,
        num_nodes, pv.max_edges
    );
}