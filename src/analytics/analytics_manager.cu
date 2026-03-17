#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable: 4996)
#endif

#include "analytics_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <cstdio>
#include <cstring>

void AnalyticsManager::init(int max_sp, int max_creatures, int history_length,
                            int heatmap_w, int heatmap_h, int phylo_max_nodes) {
    max_species = max_sp;
    density_resolution = 128;
    energy_hist_bins = 50;
    tick_counter = 0;
    export_interval = 1000;

    allocateHistoryBuffer(history, history_length);
    allocateGeneFrequencyData(gene_freq, GENOME_SIZE, 32);
    allocateReductionBuffers(reduction, 4096, max_species);
    allocateDiversityResult(diversity);
    allocatePhyloVisualizerData(phylo_viz, phylo_max_nodes);
    allocateHeatmapData(heatmap, heatmap_w, heatmap_h);
    allocateExportBuffers(export_buf, max_creatures, max_creatures, history_length);

    CUDA_CHECK(cudaMalloc(&d_species_pop, max_species * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_species_energy, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_species_fitness, max_species * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_density_grid, density_resolution * density_resolution * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_energy_hist, energy_hist_bins * sizeof(int)));

    CUDA_CHECK(cudaStreamCreate(&stream_stats));
    CUDA_CHECK(cudaStreamCreate(&stream_histogram));
    CUDA_CHECK(cudaStreamCreate(&stream_diversity));
    CUDA_CHECK(cudaStreamCreate(&stream_export));

    memset(&current_snapshot, 0, sizeof(AnalyticsSnapshot));
}

void AnalyticsManager::destroy() {
    freeHistoryBuffer(history);
    freeGeneFrequencyData(gene_freq);
    freeReductionBuffers(reduction);
    freeDiversityResult(diversity);
    freePhyloVisualizerData(phylo_viz);
    freeHeatmapData(heatmap);
    freeExportBuffers(export_buf);

    cudaFree(d_species_pop);
    cudaFree(d_species_energy);
    cudaFree(d_species_fitness);
    cudaFree(d_density_grid);
    cudaFree(d_energy_hist);

    cudaStreamDestroy(stream_stats);
    cudaStreamDestroy(stream_histogram);
    cudaStreamDestroy(stream_diversity);
    cudaStreamDestroy(stream_export);
}

__global__ void storeSnapshotKernel(
    AnalyticsSnapshot* __restrict__ history,
    int* __restrict__ write_index,
    int max_history,
    const float* __restrict__ energy_sum,
    const float* __restrict__ health_sum,
    const float* __restrict__ age_sum,
    const int* __restrict__ alive_count,
    const int* __restrict__ unique_species,
    const float* __restrict__ shannon,
    const float* __restrict__ simpson,
    int tick
) {
    if (threadIdx.x != 0) return;

    int wi = *write_index;
    int idx = wi % max_history;

    int count = *alive_count;

    history[idx].tick = tick;
    history[idx].total_alive = count;
    history[idx].total_species = *unique_species;

    if (count > 0) {
        float fc = (float)count;
        history[idx].avg_energy = *energy_sum / fc;
        history[idx].avg_health = *health_sum / fc;
        history[idx].avg_age = *age_sum / fc;
    } else {
        history[idx].avg_energy = 0.0f;
        history[idx].avg_health = 0.0f;
        history[idx].avg_age = 0.0f;
    }

    history[idx].shannon_diversity = *shannon;
    history[idx].simpson_diversity = *simpson;

    *write_index = wi + 1;
}

void AnalyticsManager::update(
    const CreatureData& creatures,
    int num_creatures,
    int current_tick,
    const PopulationStats& pop_stats
) {
    tick_counter = current_tick;

    launchComputeBasicStats(
        creatures, reduction, nullptr,
        num_creatures, stream_stats
    );

    launchComputeSpeciesCount(
        creatures, reduction,
        num_creatures, max_species, stream_stats
    );

    launchPerSpeciesStats(
        creatures, d_species_pop, d_species_energy, d_species_fitness,
        num_creatures, max_species, stream_stats
    );

    int total_alive_h = 0;
    CUDA_CHECK(cudaMemcpyAsync(&total_alive_h, reduction.d_alive_count,
        sizeof(int), cudaMemcpyDeviceToHost, stream_stats));
    CUDA_CHECK(cudaStreamSynchronize(stream_stats));

    launchShannonDiversity(
        d_species_pop, max_species, total_alive_h,
        diversity, stream_diversity
    );

    launchSimpsonDiversity(
        d_species_pop, max_species, total_alive_h,
        diversity, stream_diversity
    );

    if (current_tick % 10 == 0) {
        launchGeneHistogram(gene_freq, creatures, num_creatures, stream_histogram);
        launchNormalizeHistogram(gene_freq, total_alive_h, stream_histogram);
    }

    if (current_tick % 5 == 0) {
        launchSpatialDensityHistogram(
            creatures, d_density_grid, density_resolution,
            WORLD_SIZE, num_creatures, stream_histogram
        );
    }

    if (current_tick % 10 == 0) {
        launchEnergyDistributionHistogram(
            creatures, d_energy_hist, energy_hist_bins,
            MAX_ENERGY, num_creatures, stream_histogram
        );
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_diversity));

    storeSnapshotKernel<<<1, 1, 0, stream_stats>>>(
        history.d_history, history.d_write_index, history.max_history,
        reduction.d_energy_sum, reduction.d_health_sum,
        reduction.d_age_sum, reduction.d_alive_count,
        reduction.d_unique_species_count,
        diversity.d_shannon, diversity.d_simpson,
        current_tick
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_stats));
    CUDA_CHECK(cudaStreamSynchronize(stream_histogram));

    current_snapshot.tick = current_tick;
    current_snapshot.total_alive = total_alive_h;

    float shannon_h, simpson_h;
    int species_h;
    CUDA_CHECK(cudaMemcpy(&shannon_h, diversity.d_shannon, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&simpson_h, diversity.d_simpson, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&species_h, reduction.d_unique_species_count, sizeof(int), cudaMemcpyDeviceToHost));

    float energy_h, health_h, age_h;
    CUDA_CHECK(cudaMemcpy(&energy_h, reduction.d_energy_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&health_h, reduction.d_health_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&age_h, reduction.d_age_sum, sizeof(float), cudaMemcpyDeviceToHost));

    current_snapshot.total_species = species_h;
    current_snapshot.shannon_diversity = shannon_h;
    current_snapshot.simpson_diversity = simpson_h;

    if (total_alive_h > 0) {
        current_snapshot.avg_energy = energy_h / (float)total_alive_h;
        current_snapshot.avg_health = health_h / (float)total_alive_h;
        current_snapshot.avg_age = age_h / (float)total_alive_h;
    }
}

void AnalyticsManager::computeHeatmap(
    const float* d_scalar_field,
    int field_w, int field_h,
    float min_val, float max_val,
    HeatmapType type
) {
    launchScalarToHeatmap(heatmap, d_scalar_field,
        field_w, field_h, min_val, max_val, type, 0);
    launchGaussianBlurHeatmap(heatmap, 2, 1.5f, 0);
}

void AnalyticsManager::updatePhyloVisualization(
    const PhyloTreeGPU& tree,
    const int* d_species_pop_count,
    int num_tree_nodes
) {
    launchBuildPhyloRenderNodes(phylo_viz, tree, d_species_pop_count, max_species, 0);
    launchForceDirectedLayout(phylo_viz, num_tree_nodes, 10, 50.0f, 0.1f, 0.85f, 0);
    launchBuildEdgeBuffer(phylo_viz, num_tree_nodes, 0);
}

void AnalyticsManager::exportData(
    const CreatureData& creatures,
    int num_creatures,
    const char* output_dir
) {
    char path[512];

    snprintf(path, sizeof(path), "%s/creatures_tick_%d.csv", output_dir, tick_counter);
    asyncExportCreatureData(creatures, export_buf, num_creatures, path, stream_export);

    int wi;
    CUDA_CHECK(cudaMemcpy(&wi, history.d_write_index, sizeof(int), cudaMemcpyDeviceToHost));
    snprintf(path, sizeof(path), "%s/population_history.csv", output_dir);
    asyncExportPopulationTimeSeries(history, export_buf, wi, path, stream_export);

    snprintf(path, sizeof(path), "%s/gene_freq_tick_%d.csv", output_dir, tick_counter);
    asyncExportGeneFrequencies(gene_freq, export_buf, path, stream_export);

    snprintf(path, sizeof(path), "%s/snapshot_log.csv", output_dir);
    exportSnapshotToCSV(current_snapshot, path, (tick_counter > 0));
}

AnalyticsSnapshot AnalyticsManager::getSnapshot() {
    return current_snapshot;
}