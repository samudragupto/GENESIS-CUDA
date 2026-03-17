#ifndef DATA_EXPORTER_CUH
#define DATA_EXPORTER_CUH

#include <cuda_runtime.h>
#include "analytics_common.cuh"
#include "../creatures/creature_common.cuh"
#include "../ecosystem/ecosystem_common.cuh"

struct ExportBuffers {
    float* h_pinned_float_buf;
    int*   h_pinned_int_buf;
    AnalyticsSnapshot* h_pinned_snapshots;
    int    float_buf_size;
    int    int_buf_size;
    int    snapshot_buf_size;
};

void allocateExportBuffers(ExportBuffers& eb, int float_size, int int_size, int snap_size);
void freeExportBuffers(ExportBuffers& eb);

void asyncExportCreatureData(
    const CreatureData& creatures,
    ExportBuffers& eb,
    int num_creatures,
    const char* filename,
    cudaStream_t stream = 0
);

void asyncExportPopulationTimeSeries(
    const HistoryBuffer& history,
    ExportBuffers& eb,
    int num_entries,
    const char* filename,
    cudaStream_t stream = 0
);

void asyncExportGeneFrequencies(
    const GeneFrequencyData& gene_freq,
    ExportBuffers& eb,
    const char* filename,
    cudaStream_t stream = 0
);

void exportSnapshotToCSV(
    const AnalyticsSnapshot& snapshot,
    const char* filename,
    bool append
);

void exportSpeciesDataToCSV(
    const int* h_species_pop,
    const float* h_species_energy,
    int max_species,
    const char* filename
);

#endif