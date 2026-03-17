#ifndef BENCHMARK_SUITE_CUH
#define BENCHMARK_SUITE_CUH

#include <cuda_runtime.h>

struct BenchmarkResult {
    char name[128];
    float avg_ms;
    float min_ms;
    float max_ms;
    float stddev_ms;
    float throughput;
    char throughput_unit[32];
    int iterations;
};

#define MAX_BENCHMARKS 64

class BenchmarkSuite {
public:
    BenchmarkResult results[MAX_BENCHMARKS];
    int num_results;
    int warmup_iterations;
    int benchmark_iterations;

    void init(int warmup = 5, int iterations = 100);
    void destroy();

    void benchmarkSpatialHash(int num_particles, int grid_size);
    void benchmarkSPHDensity(int num_particles);
    void benchmarkNeuralForward(int num_creatures);
    void benchmarkGenetics(int num_creatures);
    void benchmarkTerrainGen(int world_size);
    void benchmarkClimate(int world_size);
    void benchmarkVegetation(int world_size);
    void benchmarkStreamCompaction(int num_elements);
    void benchmarkReduction(int num_elements);
    void benchmarkHaloExchange(int world_size, int halo_width);
    void benchmarkMemoryBandwidth(size_t bytes);

    void runAll(int world_size, int num_creatures);

    void addResult(const char* name, float* times, int count,
                   float throughput = 0.0f, const char* unit = "");

    void printResults() const;
    void exportCSV(const char* filename) const;
};

#endif