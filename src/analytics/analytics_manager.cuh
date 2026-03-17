#ifndef ANALYTICS_MANAGER_CUH
#define ANALYTICS_MANAGER_CUH

#include <cuda_runtime.h>
#include "analytics_common.cuh"
#include "parallel_statistics.cuh"
#include "histogram_kernel.cuh"
#include "diversity_index.cuh"
#include "phylo_visualizer.cuh"
#include "heatmap_generator.cuh"
#include "data_exporter.cuh"
#include "../creatures/creature_common.cuh"
#include "../ecosystem/ecosystem_common.cuh"
#include "../genetics/phylogenetic_tree.cuh"

class AnalyticsManager {
public:
    HistoryBuffer history;
    GeneFrequencyData gene_freq;
    ReductionBuffers reduction;
    DiversityResult diversity;
    PhyloVisualizerData phylo_viz;
    HeatmapData heatmap;
    ExportBuffers export_buf;

    int* d_species_pop;
    float* d_species_energy;
    float* d_species_fitness;
    float* d_density_grid;
    int* d_energy_hist;

    int max_species;
    int density_resolution;
    int energy_hist_bins;
    int tick_counter;
    int export_interval;

    cudaStream_t stream_stats;
    cudaStream_t stream_histogram;
    cudaStream_t stream_diversity;
    cudaStream_t stream_export;

    AnalyticsSnapshot current_snapshot;

    void init(int max_species, int max_creatures, int history_length,
              int heatmap_w, int heatmap_h, int phylo_max_nodes);
    void destroy();

    void update(
        const CreatureData& creatures,
        int num_creatures,
        int current_tick,
        const PopulationStats& pop_stats
    );

    void computeHeatmap(
        const float* d_scalar_field,
        int field_w, int field_h,
        float min_val, float max_val,
        HeatmapType type
    );

    void updatePhyloVisualization(
        const PhyloTreeGPU& tree,
        const int* d_species_pop_count,
        int num_tree_nodes
    );

    void exportData(
        const CreatureData& creatures,
        int num_creatures,
        const char* output_dir
    );

    AnalyticsSnapshot getSnapshot();
};

#endif