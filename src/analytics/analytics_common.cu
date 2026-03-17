#include "analytics_common.cuh"
#include "../core/cuda_utils.cuh"

void allocateHistoryBuffer(HistoryBuffer& buf, int max_history) {
    buf.max_history = max_history;
    CUDA_CHECK(cudaMalloc(&buf.d_history, max_history * sizeof(AnalyticsSnapshot)));
    CUDA_CHECK(cudaMalloc(&buf.d_write_index, sizeof(int)));
    CUDA_CHECK(cudaMemset(buf.d_history, 0, max_history * sizeof(AnalyticsSnapshot)));
    CUDA_CHECK(cudaMemset(buf.d_write_index, 0, sizeof(int)));
}

void freeHistoryBuffer(HistoryBuffer& buf) {
    cudaFree(buf.d_history);
    cudaFree(buf.d_write_index);
}

void allocateGeneFrequencyData(GeneFrequencyData& gf, int num_genes, int num_bins) {
    gf.num_genes = num_genes;
    gf.num_bins = num_bins;
    CUDA_CHECK(cudaMalloc(&gf.d_gene_histograms, num_genes * num_bins * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gf.d_gene_counts, num_genes * num_bins * sizeof(int)));
    CUDA_CHECK(cudaMemset(gf.d_gene_histograms, 0, num_genes * num_bins * sizeof(float)));
    CUDA_CHECK(cudaMemset(gf.d_gene_counts, 0, num_genes * num_bins * sizeof(int)));
}

void freeGeneFrequencyData(GeneFrequencyData& gf) {
    cudaFree(gf.d_gene_histograms);
    cudaFree(gf.d_gene_counts);
}