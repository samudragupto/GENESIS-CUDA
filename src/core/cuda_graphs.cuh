#ifndef CUDA_GRAPHS_CUH
#define CUDA_GRAPHS_CUH

#include <cuda_runtime.h>

#define MAX_GRAPH_NODES 256

struct GraphNodeInfo {
    cudaGraphNode_t node;
    char name[64];
    int active;
};

class CudaGraphManager {
public:
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t capture_stream;
    int is_captured;
    int is_valid;
    int node_count;
    GraphNodeInfo nodes[MAX_GRAPH_NODES];

    void init();
    void destroy();

    void beginCapture(cudaStream_t stream);
    void endCapture(cudaStream_t stream);

    void launch(cudaStream_t stream);

    void invalidate();
    bool isValid() const;

    void updateKernelNode(int node_index, void* kernel_func,
                          dim3 grid, dim3 block, void** args,
                          size_t shared_mem);

    void printGraphInfo() const;
};

class GraphCaptureScope {
public:
    CudaGraphManager* manager;
    cudaStream_t stream;

    GraphCaptureScope(CudaGraphManager* mgr, cudaStream_t s);
    ~GraphCaptureScope();
};

#endif