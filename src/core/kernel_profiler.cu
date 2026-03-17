#include "kernel_profiler.cuh"
#include "cuda_utils.cuh"
#include <cstdio>
#include <cstring>
#include <algorithm>

void KernelProfiler::init() {
    num_profiles = 0;
    current_recording = -1;
    enabled = 1;
    memset(profiles, 0, sizeof(profiles));

    for (int i = 0; i < MAX_PROFILED_KERNELS; i++) {
        CUDA_CHECK(cudaEventCreate(&start_events[i]));
        CUDA_CHECK(cudaEventCreate(&stop_events[i]));
    }
}

void KernelProfiler::destroy() {
    for (int i = 0; i < MAX_PROFILED_KERNELS; i++) {
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(stop_events[i]);
    }
}

int KernelProfiler::registerKernel(const char* name) {
    if (num_profiles >= MAX_PROFILED_KERNELS) return -1;

    for (int i = 0; i < num_profiles; i++) {
        if (strcmp(profiles[i].name, name) == 0) return i;
    }

    int id = num_profiles++;
    strncpy(profiles[id].name, name, MAX_KERNEL_NAME_LEN - 1);
    profiles[id].name[MAX_KERNEL_NAME_LEN - 1] = '\0';
    profiles[id].total_ms = 0.0f;
    profiles[id].min_ms = 1e30f;
    profiles[id].max_ms = 0.0f;
    profiles[id].avg_ms = 0.0f;
    profiles[id].last_ms = 0.0f;
    profiles[id].history_index = 0;
    profiles[id].call_count = 0;
    profiles[id].active = 0;
    memset(profiles[id].history, 0, sizeof(profiles[id].history));
    return id;
}

void KernelProfiler::beginProfile(int kernel_id, cudaStream_t stream) {
    if (!enabled || kernel_id < 0 || kernel_id >= num_profiles) return;
    profiles[kernel_id].active = 1;
    CUDA_CHECK(cudaEventRecord(start_events[kernel_id], stream));
}

void KernelProfiler::endProfile(int kernel_id, cudaStream_t stream) {
    if (!enabled || kernel_id < 0 || kernel_id >= num_profiles) return;
    CUDA_CHECK(cudaEventRecord(stop_events[kernel_id], stream));
}

void KernelProfiler::collectResults() {
    if (!enabled) return;

    for (int i = 0; i < num_profiles; i++) {
        if (!profiles[i].active) continue;

        float ms = 0.0f;
        cudaError_t err = cudaEventElapsedTime(&ms, start_events[i], stop_events[i]);
        if (err != cudaSuccess) continue;

        profiles[i].last_ms = ms;
        profiles[i].total_ms += ms;
        profiles[i].call_count++;

        if (ms < profiles[i].min_ms) profiles[i].min_ms = ms;
        if (ms > profiles[i].max_ms) profiles[i].max_ms = ms;
        profiles[i].avg_ms = profiles[i].total_ms / (float)profiles[i].call_count;

        int hi = profiles[i].history_index % PROFILE_HISTORY_LEN;
        profiles[i].history[hi] = ms;
        profiles[i].history_index++;

        profiles[i].active = 0;
    }
}

KernelProfile KernelProfiler::getProfile(int kernel_id) const {
    if (kernel_id < 0 || kernel_id >= num_profiles) {
        KernelProfile empty;
        memset(&empty, 0, sizeof(empty));
        return empty;
    }
    return profiles[kernel_id];
}

void KernelProfiler::printReport() const {
    printf("\n========== KERNEL PROFILER REPORT ==========\n");
    printf("%-30s %8s %8s %8s %8s %8s\n",
           "Kernel", "Calls", "Last(ms)", "Avg(ms)", "Min(ms)", "Max(ms)");
    printf("--------------------------------------------------------------\n");

    for (int i = 0; i < num_profiles; i++) {
        if (profiles[i].call_count == 0) continue;
        printf("%-30s %8d %8.3f %8.3f %8.3f %8.3f\n",
               profiles[i].name,
               profiles[i].call_count,
               profiles[i].last_ms,
               profiles[i].avg_ms,
               profiles[i].min_ms,
               profiles[i].max_ms);
    }

    printf("--------------------------------------------------------------\n");
    printf("Total frame time: %.3f ms\n", getTotalFrameTime());
    printf("=============================================\n\n");
}

void KernelProfiler::printTopN(int n) const {
    int indices[MAX_PROFILED_KERNELS];
    float times[MAX_PROFILED_KERNELS];
    int count = 0;

    for (int i = 0; i < num_profiles; i++) {
        if (profiles[i].call_count > 0) {
            indices[count] = i;
            times[count] = profiles[i].avg_ms;
            count++;
        }
    }

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (times[j] > times[i]) {
                float tmp_t = times[i]; times[i] = times[j]; times[j] = tmp_t;
                int tmp_i = indices[i]; indices[i] = indices[j]; indices[j] = tmp_i;
            }
        }
    }

    int show = (n < count) ? n : count;
    printf("\n=== Top %d Kernels by Avg Time ===\n", show);
    for (int i = 0; i < show; i++) {
        printf("  %d. %-30s  %.3f ms (%.1f%%)\n",
               i + 1, profiles[indices[i]].name, times[i],
               (getTotalFrameTime() > 0.0f) ? times[i] / getTotalFrameTime() * 100.0f : 0.0f);
    }
}

void KernelProfiler::resetAll() {
    for (int i = 0; i < num_profiles; i++) {
        profiles[i].total_ms = 0.0f;
        profiles[i].min_ms = 1e30f;
        profiles[i].max_ms = 0.0f;
        profiles[i].avg_ms = 0.0f;
        profiles[i].last_ms = 0.0f;
        profiles[i].call_count = 0;
        profiles[i].history_index = 0;
        memset(profiles[i].history, 0, sizeof(profiles[i].history));
    }
}

float KernelProfiler::getTotalFrameTime() const {
    float total = 0.0f;
    for (int i = 0; i < num_profiles; i++) {
        total += profiles[i].last_ms;
    }
    return total;
}

float KernelProfiler::getKernelTime(const char* name) const {
    for (int i = 0; i < num_profiles; i++) {
        if (strcmp(profiles[i].name, name) == 0) {
            return profiles[i].last_ms;
        }
    }
    return 0.0f;
}

template<typename KernelFunc>
OccupancyInfo KernelProfiler::queryOccupancy(KernelFunc func, int block_size, int shared_mem) {
    OccupancyInfo info;
    info.block_size = block_size;
    info.shared_mem_bytes = shared_mem;

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info.grid_size, func, block_size, shared_mem);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    info.max_warps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    info.active_warps = info.grid_size * (block_size / prop.warpSize);
    info.occupancy = (float)info.active_warps / (float)info.max_warps;

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, func);
    info.registers_per_thread = attr.numRegs;

    return info;
}

ScopedProfiler::ScopedProfiler(KernelProfiler* p, int id, cudaStream_t s)
    : profiler(p), kernel_id(id), stream(s) {
    if (profiler) profiler->beginProfile(kernel_id, stream);
}

ScopedProfiler::~ScopedProfiler() {
    if (profiler) profiler->endProfile(kernel_id, stream);
}