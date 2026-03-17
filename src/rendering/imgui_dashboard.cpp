#include "imgui_dashboard.h"
#include <cstring>
#include <cstdio>

void ImGuiDashboard::init() {
    memset(&state, 0, sizeof(DashboardState));
    state.show_stats = true;
    state.show_population = true;
    state.show_controls = true;
    state.show_performance = true;
    state.show_gene_freq = false;
    state.show_heatmap_selector = false;
    state.show_phylo_tree = false;
    state.selected_heatmap = 0;
    state.simulation_speed = 1.0f;
    state.paused = false;
    state.fps_index = 0;
    state.pop_index = 0;
    state.species_index = 0;
    state.energy_index = 0;
}

void ImGuiDashboard::destroy() {
}

void ImGuiDashboard::beginFrame() {
}

void ImGuiDashboard::endFrame() {
}

void ImGuiDashboard::renderStatsWindow(const AnalyticsSnapshot& snapshot) {
    printf("=== GENESIS Stats (Tick %d) ===\n", snapshot.tick);
    printf("  Alive: %d | Species: %d\n", snapshot.total_alive, snapshot.total_species);
    printf("  Avg Energy: %.3f | Avg Health: %.3f | Avg Age: %.1f\n",
           snapshot.avg_energy, snapshot.avg_health, snapshot.avg_age);
    printf("  Shannon: %.3f | Simpson: %.3f\n",
           snapshot.shannon_diversity, snapshot.simpson_diversity);
}

void ImGuiDashboard::renderPopulationGraph(const AnalyticsSnapshot& snapshot) {
    addPopulationSample(snapshot.total_alive);
    addSpeciesSample(snapshot.total_species);
    addEnergySample(snapshot.avg_energy);
}

void ImGuiDashboard::renderControlPanel(float& sim_speed, bool& paused,
                                        bool& trigger_disease, int& heatmap_type,
                                        bool& export_data) {
    sim_speed = state.simulation_speed;
    paused = state.paused;
    trigger_disease = false;
    heatmap_type = state.selected_heatmap;
    export_data = false;
}

void ImGuiDashboard::renderPerformanceWindow(float frame_time, float sim_time,
                                              float render_time) {
    float fps = (frame_time > 0.0f) ? 1000.0f / frame_time : 0.0f;
    addFPSSample(fps);
    printf("  FPS: %.1f | Frame: %.2fms | Sim: %.2fms | Render: %.2fms\n",
           fps, frame_time, sim_time, render_time);
}

void ImGuiDashboard::renderGeneFrequencyWindow(const float* gene_data,
                                                int num_genes, int num_bins) {
    if (!gene_data) return;
}

void ImGuiDashboard::renderHeatmapSelector(int& selected) {
    selected = state.selected_heatmap;
}

void ImGuiDashboard::renderPhyloTreeWindow() {
}

void ImGuiDashboard::addFPSSample(float fps) {
    state.fps_history[state.fps_index % 120] = fps;
    state.fps_index++;
}

void ImGuiDashboard::addPopulationSample(int alive) {
    state.pop_history[state.pop_index % 300] = (float)alive;
    state.pop_index++;
}

void ImGuiDashboard::addSpeciesSample(int species) {
    state.species_history[state.species_index % 300] = (float)species;
    state.species_index++;
}

void ImGuiDashboard::addEnergySample(float energy) {
    state.energy_history[state.energy_index % 300] = energy;
    state.energy_index++;
}