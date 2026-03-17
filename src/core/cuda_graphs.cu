#include "cuda_graphs.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstring>

void CudaGraphManager::init() {
    graph = nullptr;
    graph_exec = nullptr;
    capture_stream = nullptr;
    is_captured = 0;
    is_valid = 0;
    node_count = 0;
    memset(nodes, 0, sizeof(nodes));
}

void CudaGraphManager::destroy() {
    if (graph_exec) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
    }
    if (graph) {
        cudaGraphDestroy(graph);
        graph = nullptr;
    }
    is_captured = 0;
    is_valid = 0;
}

void CudaGraphManager::beginCapture(cudaStream_t stream) {
    if (graph_exec) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
    }
    if (graph) {
        cudaGraphDestroy(graph);
        graph = nullptr;
    }

    capture_stream = stream;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    is_captured = 0;
    is_valid = 0;
    node_count = 0;
}

void CudaGraphManager::endCapture(cudaStream_t stream) {
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));

    if (graph) {
        size_t num_nodes = 0;
        CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
        node_count = (int)num_nodes;

        CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        is_captured = 1;
        is_valid = 1;
    }
}

void CudaGraphManager::launch(cudaStream_t stream) {
    if (!is_valid || !graph_exec) return;
    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
}

void CudaGraphManager::invalidate() {
    is_valid = 0;
}

bool CudaGraphManager::isValid() const {
    return is_valid != 0;
}

void CudaGraphManager::updateKernelNode(int node_index, void* kernel_func,
                                          dim3 grid, dim3 block, void** args,
                                          size_t shared_mem) {
    if (!graph || !graph_exec) return;

    size_t num_nodes = 0;
    CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));

    if (node_index < 0 || node_index >= (int)num_nodes) return;

    cudaGraphNode_t* graph_nodes = new cudaGraphNode_t[num_nodes];
    CUDA_CHECK(cudaGraphGetNodes(graph, graph_nodes, &num_nodes));

    cudaKernelNodeParams node_params;
    memset(&node_params, 0, sizeof(node_params));
    node_params.func = kernel_func;
    node_params.gridDim = grid;
    node_params.blockDim = block;
    node_params.kernelParams = args;
    node_params.sharedMemBytes = shared_mem;

    cudaGraphExecKernelNodeSetParams(graph_exec, graph_nodes[node_index], &node_params);

    delete[] graph_nodes;
}

void CudaGraphManager::printGraphInfo() const {
    printf("CUDA Graph Info:\n");
    printf("  Captured: %s\n", is_captured ? "Yes" : "No");
    printf("  Valid: %s\n", is_valid ? "Yes" : "No");
    printf("  Nodes: %d\n", node_count);
}

GraphCaptureScope::GraphCaptureScope(CudaGraphManager* mgr, cudaStream_t s)
    : manager(mgr), stream(s) {
    if (manager) manager->beginCapture(stream);
}

GraphCaptureScope::~GraphCaptureScope() {
    if (manager) manager->endCapture(stream);
}