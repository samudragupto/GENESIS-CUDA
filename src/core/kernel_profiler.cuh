#ifndef KERNEL_PROFILER_CUH
#define KERNEL_PROFILER_CUH

#include <cuda_runtime.h>

#define MAX_PROFILED_KERNELS 128
#define MAX_KERNEL_NAME_LEN 64
#define PROFILE_HISTORY_LEN 256

struct KernelProfile {
    char name[MAX_KERNEL_NAME_LEN];
    float total_ms;
    float min_ms;
    float max_ms;
    float avg_ms;
    float last_ms;
    float history[PROFILE_HISTORY_LEN];
    int history_index;
    int call_count;
    int active;
};

struct OccupancyInfo {
    int block_size;
    int grid_size;
    int active_warps;
    int max_warps;
    float occupancy;
    int shared_mem_bytes;
    int registers_per_thread;
};

class KernelProfiler {
public:
    KernelProfile profiles[MAX_PROFILED_KERNELS];
    int num_profiles;
    cudaEvent_t start_events[MAX_PROFILED_KERNELS];
    cudaEvent_t stop_events[MAX_PROFILED_KERNELS];
    int current_recording;
    int enabled;

    void init();
    void destroy();

    int registerKernel(const char* name);
    void beginProfile(int kernel_id, cudaStream_t stream = 0);
    void endProfile(int kernel_id, cudaStream_t stream = 0);
    void collectResults();

    KernelProfile getProfile(int kernel_id) const;
    void printReport() const;
    void printTopN(int n) const;
    void resetAll();

    float getTotalFrameTime() const;
    float getKernelTime(const char* name) const;

    template<typename KernelFunc>
    OccupancyInfo queryOccupancy(KernelFunc func, int block_size, int shared_mem);
};

class ScopedProfiler {
public:
    KernelProfiler* profiler;
    int kernel_id;
    cudaStream_t stream;

    ScopedProfiler(KernelProfiler* p, int id, cudaStream_t s = 0);
    ~ScopedProfiler();
};

#define PROFILE_KERNEL(profiler, id, stream) ScopedProfiler _sp_##id(&profiler, id, stream)

#endif