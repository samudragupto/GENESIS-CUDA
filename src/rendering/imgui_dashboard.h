#ifndef IMGUI_DASHBOARD_H
#define IMGUI_DASHBOARD_H

#include "../analytics/analytics_common.cuh"
#include "../ecosystem/ecosystem_common.cuh"

struct DashboardState {
    bool show_stats;
    bool show_population;
    bool show_gene_freq;
    bool show_controls;
    bool show_heatmap_selector;
    bool show_phylo_tree;
    bool show_performance;

    int selected_heatmap;
    float simulation_speed;
    bool paused;

    float fps_history[120];
    int fps_index;

    float pop_history[300];
    int pop_index;

    float species_history[300];
    int species_index;

    float energy_history[300];
    int energy_index;
};

class ImGuiDashboard {
public:
    DashboardState state;

    void init();
    void destroy();

    void beginFrame();
    void endFrame();

    void renderStatsWindow(const AnalyticsSnapshot& snapshot);
    void renderPopulationGraph(const AnalyticsSnapshot& snapshot);
    void renderControlPanel(float& sim_speed, bool& paused, bool& trigger_disease,
                           int& heatmap_type, bool& export_data);
    void renderPerformanceWindow(float frame_time, float sim_time, float render_time);
    void renderGeneFrequencyWindow(const float* gene_data, int num_genes, int num_bins);
    void renderHeatmapSelector(int& selected);
    void renderPhyloTreeWindow();

    void addFPSSample(float fps);
    void addPopulationSample(int alive);
    void addSpeciesSample(int species);
    void addEnergySample(float energy);
};

#endif