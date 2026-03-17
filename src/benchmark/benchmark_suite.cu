#include "benchmark_suite.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <cstdio>
#include <cstring>
#include <cmath>

void BenchmarkSuite::init(int warmup, int iterations) {
    warmup_iterations = warmup;
    benchmark_iterations = iterations;
    num_results = 0;
    memset(results, 0, sizeof(results));
}

void BenchmarkSuite::destroy() {
}

void BenchmarkSuite::addResult(const char* name, float* times, int count,
                                float throughput, const char* unit) {
    if (num_results >= MAX_BENCHMARKS) return;

    BenchmarkResult& r = results[num_results++];
    strncpy(r.name, name, 127);
    r.iterations = count;

    float sum = 0.0f;
    r.min_ms = 1e30f;
    r.max_ms = 0.0f;

    for (int i = 0; i < count; i++) {
        sum += times[i];
        if (times[i] < r.min_ms) r.min_ms = times[i];
        if (times[i] > r.max_ms) r.max_ms = times[i];
    }
    r.avg_ms = sum / (float)count;

    float var = 0.0f;
    for (int i = 0; i < count; i++) {
        float d = times[i] - r.avg_ms;
        var += d * d;
    }
    r.stddev_ms = sqrtf(var / (float)count);

    r.throughput = throughput;
    strncpy(r.throughput_unit, unit, 31);
}

__global__ void benchmarkCopyKernel(float* __restrict__ dst,
                                     const float* __restrict__ src, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    dst[idx] = src[idx];
}

__global__ void benchmarkReduceKernel(const float* __restrict__ input,
                                       float* __restrict__ output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}

void BenchmarkSuite::benchmarkMemoryBandwidth(size_t bytes) {
    int n = (int)(bytes / sizeof(float));
    float* d_src;
    float* d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMemset(d_src, 0, bytes));

    int block = 256;
    int grid = (n + block - 1) / block;

    for (int i = 0; i < warmup_iterations; i++) {
        benchmarkCopyKernel<<<grid, block>>>(d_dst, d_src, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float* times = new float[benchmark_iterations];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < benchmark_iterations; i++) {
        CUDA_CHECK(cudaEventRecord(start));
        benchmarkCopyKernel<<<grid, block>>>(d_dst, d_src, n);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    float avg_ms = 0.0f;
    for (int i = 0; i < benchmark_iterations; i++) avg_ms += times[i];
    avg_ms /= (float)benchmark_iterations;

    float bw = (float)(bytes * 2) / (avg_ms * 1e-3f) / 1e9f;
    addResult("Memory Bandwidth (Copy)", times, benchmark_iterations, bw, "GB/s");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] times;
    cudaFree(d_src);
    cudaFree(d_dst);
}

void BenchmarkSuite::benchmarkReduction(int num_elements) {
    float* d_input;
    float* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, num_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

    float* h_input = new float[num_elements];
    for (int i = 0; i < num_elements; i++) h_input[i] = 1.0f;
    CUDA_CHECK(cudaMemcpy(d_input, h_input, num_elements * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (num_elements + block - 1) / block;

    for (int i = 0; i < warmup_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        benchmarkReduceKernel<<<grid, block>>>(d_input, d_output, num_elements);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    float* times = new float[benchmark_iterations];
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < benchmark_iterations; i++) {
        CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));
        CUDA_CHECK(cudaEventRecord(start));
        benchmarkReduceKernel<<<grid, block>>>(d_input, d_output, num_elements);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[i], start, stop));
    }

    float throughput = (float)num_elements / 1e6f;
    addResult("Parallel Reduction", times, benchmark_iterations, throughput, "M elements");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    delete[] times;
    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
}

void BenchmarkSuite::benchmarkSpatialHash(int num_particles, int grid_size) {
    printf("  [Benchmark] Spatial Hash: %d particles, %d grid\n", num_particles, grid_size);
}

void BenchmarkSuite::benchmarkSPHDensity(int num_particles) {
    printf("  [Benchmark] SPH Density: %d particles\n", num_particles);
}

void BenchmarkSuite::benchmarkNeuralForward(int num_creatures) {
    printf("  [Benchmark] Neural Forward: %d creatures\n", num_creatures);
}

void BenchmarkSuite::benchmarkGenetics(int num_creatures) {
    printf("  [Benchmark] Genetics Pipeline: %d creatures\n", num_creatures);
}

void BenchmarkSuite::benchmarkTerrainGen(int world_size) {
    printf("  [Benchmark] Terrain Generation: %dx%d\n", world_size, world_size);
}

void BenchmarkSuite::benchmarkClimate(int world_size) {
    printf("  [Benchmark] Climate Simulation: %dx%d\n", world_size, world_size);
}

void BenchmarkSuite::benchmarkVegetation(int world_size) {
    printf("  [Benchmark] Vegetation Update: %dx%d\n", world_size, world_size);
}

void BenchmarkSuite::benchmarkStreamCompaction(int num_elements) {
    printf("  [Benchmark] Stream Compaction: %d elements\n", num_elements);
}

void BenchmarkSuite::benchmarkHaloExchange(int world_size, int halo_width) {
    printf("  [Benchmark] Halo Exchange: %dx%d, halo=%d\n", world_size, world_size, halo_width);
}

void BenchmarkSuite::runAll(int world_size, int num_creatures) {
    printf("\n============ GENESIS Benchmark Suite ============\n");

    benchmarkMemoryBandwidth(256 * 1024 * 1024);
    benchmarkReduction(num_creatures);
    benchmarkSpatialHash(num_creatures, world_size / 4);
    benchmarkNeuralForward(num_creatures);
    benchmarkGenetics(num_creatures);
    benchmarkTerrainGen(world_size);
    benchmarkClimate(world_size);
    benchmarkVegetation(world_size);
    benchmarkStreamCompaction(num_creatures);

    printf("=================================================\n");
    printResults();
}

void BenchmarkSuite::printResults() const {
    printf("\n============ Benchmark Results ============\n");
    printf("%-35s %10s %10s %10s %10s %12s\n",
           "Benchmark", "Avg(ms)", "Min(ms)", "Max(ms)", "StdDev", "Throughput");
    printf("------------------------------------------------------------------------------------\n");

    for (int i = 0; i < num_results; i++) {
        const BenchmarkResult& r = results[i];
        if (r.throughput > 0.0f) {
            printf("%-35s %10.3f %10.3f %10.3f %10.3f %8.2f %s\n",
                   r.name, r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms,
                   r.throughput, r.throughput_unit);
        } else {
            printf("%-35s %10.3f %10.3f %10.3f %10.3f\n",
                   r.name, r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms);
        }
    }
    printf("==========================================\n\n");
}

void BenchmarkSuite::exportCSV(const char* filename) const {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "name,avg_ms,min_ms,max_ms,stddev_ms,throughput,throughput_unit,iterations\n");
    for (int i = 0; i < num_results; i++) {
        const BenchmarkResult& r = results[i];
        fprintf(fp, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%s,%d\n",
                r.name, r.avg_ms, r.min_ms, r.max_ms, r.stddev_ms,
                r.throughput, r.throughput_unit, r.iterations);
    }
    fclose(fp);
}