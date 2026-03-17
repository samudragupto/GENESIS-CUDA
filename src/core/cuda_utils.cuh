#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define KERNEL_CHECK(...)                                                   \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Kernel Error at %s:%d - %s\n",           \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUDA_CHECK_LAST() KERNEL_CHECK()

inline dim3 computeGrid1D(int n, int block_size = 256) {
    return dim3((n + block_size - 1) / block_size);
}

inline dim3 computeGrid2D(int w, int h, int bx = 16, int by = 16) {
    return dim3((w + bx - 1) / bx, (h + by - 1) / by);
}

inline dim3 gridSize2D(int width, int height, int block_size) {
    return dim3((width + block_size - 1) / block_size, (height + block_size - 1) / block_size);
}

struct GpuTimer {
    cudaEvent_t start_event, stop_event;
    GpuTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
    }
    float elapsedMs() {
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

#ifdef _WIN32
    #include <direct.h>
    #define GENESIS_MKDIR(dir) _mkdir(dir)
#else
    #include <sys/stat.h>
    #define GENESIS_MKDIR(dir) mkdir(dir, 0755)
#endif

inline void ensureDirectory(const char* path) {
    GENESIS_MKDIR(path);
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif