#include "data_exporter.cuh"
#include "../core/cuda_utils.cuh"
#include <cstdio>
#include <cstring>

void allocateExportBuffers(ExportBuffers& eb, int float_size, int int_size, int snap_size) {
    eb.float_buf_size = float_size;
    eb.int_buf_size = int_size;
    eb.snapshot_buf_size = snap_size;

    CUDA_CHECK(cudaMallocHost(&eb.h_pinned_float_buf, float_size * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&eb.h_pinned_int_buf, int_size * sizeof(int)));
    CUDA_CHECK(cudaMallocHost(&eb.h_pinned_snapshots, snap_size * sizeof(AnalyticsSnapshot)));
}

void freeExportBuffers(ExportBuffers& eb) {
    cudaFreeHost(eb.h_pinned_float_buf);
    cudaFreeHost(eb.h_pinned_int_buf);
    cudaFreeHost(eb.h_pinned_snapshots);
}

struct AsyncWriteContext {
    float* data;
    int count;
    char filename[256];
};

void asyncExportCreatureData(
    const CreatureData& creatures,
    ExportBuffers& eb,
    int num_creatures,
    const char* filename,
    cudaStream_t stream
) {
    int copy_count = min(num_creatures, eb.float_buf_size);

    CUDA_CHECK(cudaMemcpyAsync(eb.h_pinned_float_buf, creatures.d_energy,
        copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "index,energy\n");
    for (int i = 0; i < copy_count; i++) {
        fprintf(fp, "%d,%.6f\n", i, eb.h_pinned_float_buf[i]);
    }
    fclose(fp);
}

void asyncExportPopulationTimeSeries(
    const HistoryBuffer& history,
    ExportBuffers& eb,
    int num_entries,
    const char* filename,
    cudaStream_t stream
) {
    int copy_count = min(num_entries, eb.snapshot_buf_size);

    CUDA_CHECK(cudaMemcpyAsync(eb.h_pinned_snapshots, history.d_history,
        copy_count * sizeof(AnalyticsSnapshot), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "tick,alive,species,avg_energy,avg_health,avg_age,shannon,simpson\n");
    for (int i = 0; i < copy_count; i++) {
        AnalyticsSnapshot& s = eb.h_pinned_snapshots[i];
        fprintf(fp, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
            s.tick, s.total_alive, s.total_species,
            s.avg_energy, s.avg_health, s.avg_age,
            s.shannon_diversity, s.simpson_diversity);
    }
    fclose(fp);
}

void asyncExportGeneFrequencies(
    const GeneFrequencyData& gene_freq,
    ExportBuffers& eb,
    const char* filename,
    cudaStream_t stream
) {
    int total = gene_freq.num_genes * gene_freq.num_bins;
    int copy_count = min(total, eb.float_buf_size);

    CUDA_CHECK(cudaMemcpyAsync(eb.h_pinned_float_buf, gene_freq.d_gene_histograms,
        copy_count * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "gene,bin,frequency\n");
    for (int g = 0; g < gene_freq.num_genes; g++) {
        for (int b = 0; b < gene_freq.num_bins; b++) {
            int idx = g * gene_freq.num_bins + b;
            if (idx < copy_count) {
                fprintf(fp, "%d,%d,%.6f\n", g, b, eb.h_pinned_float_buf[idx]);
            }
        }
    }
    fclose(fp);
}

void exportSnapshotToCSV(
    const AnalyticsSnapshot& snapshot,
    const char* filename,
    bool append
) {
    FILE* fp = fopen(filename, append ? "a" : "w");
    if (!fp) return;

    if (!append) {
        fprintf(fp, "tick,alive,species,avg_energy,avg_health,avg_age,shannon,simpson,vegetation,temperature\n");
    }

    fprintf(fp, "%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        snapshot.tick, snapshot.total_alive, snapshot.total_species,
        snapshot.avg_energy, snapshot.avg_health, snapshot.avg_age,
        snapshot.shannon_diversity, snapshot.simpson_diversity,
        snapshot.total_vegetation, snapshot.avg_temperature);

    fclose(fp);
}

void exportSpeciesDataToCSV(
    const int* h_species_pop,
    const float* h_species_energy,
    int max_species,
    const char* filename
) {
    FILE* fp = fopen(filename, "w");
    if (!fp) return;

    fprintf(fp, "species_id,population,avg_energy\n");
    for (int i = 0; i < max_species; i++) {
        if (h_species_pop[i] > 0) {
            fprintf(fp, "%d,%d,%.6f\n", i, h_species_pop[i], h_species_energy[i]);
        }
    }
    fclose(fp);
}