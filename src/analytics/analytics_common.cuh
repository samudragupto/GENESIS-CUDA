#ifndef ANALYTICS_COMMON_CUH
#define ANALYTICS_COMMON_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

struct AnalyticsSnapshot {
    int tick;
    int total_alive;
    int total_species;
    float avg_energy;
    float avg_health;
    float avg_age;
    float shannon_diversity;
    float simpson_diversity;
    float total_vegetation;
    float avg_temperature;
};

struct SpeciesSnapshot {
    int species_id;
    int population;
    float avg_energy;
    float avg_fitness;
    float avg_genome[GENOME_SIZE];
};

struct HistoryBuffer {
    AnalyticsSnapshot* d_history;
    int* d_write_index;
    int max_history;
};

struct GeneFrequencyData {
    float* d_gene_histograms;
    int num_bins;
    int num_genes;
    int* d_gene_counts;
};

void allocateHistoryBuffer(HistoryBuffer& buf, int max_history);
void freeHistoryBuffer(HistoryBuffer& buf);
void allocateGeneFrequencyData(GeneFrequencyData& gf, int num_genes, int num_bins);
void freeGeneFrequencyData(GeneFrequencyData& gf);

#endif