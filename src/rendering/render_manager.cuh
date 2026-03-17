#ifndef RENDER_MANAGER_CUH
#define RENDER_MANAGER_CUH

#include <cuda_runtime.h>
#include "render_common.cuh"
#include "atmosphere_renderer.cuh"
#include "cloud_raymarcher.cuh"
#include "day_night_cycle.cuh"
#include "post_processing.cuh"
#include "camera_controller.h"
#include "imgui_dashboard.h"
#include "../analytics/analytics_common.cuh"
#include "../analytics/heatmap_generator.cuh"
#include "../terrain/terrain_renderer.cuh"

class RenderManager {
public:
    FrameBuffer framebuffer;
    PostProcessBuffers post_buffers;
    PostProcessParams post_params;
    AtmosphereParams atmo_params;
    CloudParams cloud_params;
    LightParams light;
    DayNightState day_night;
    CameraController camera_ctrl;
    ImGuiDashboard dashboard;
    HeatmapData heatmap;

    TerrainMeshData terrain_mesh;
    TerrainRenderConfig terrain_config;
    int terrain_initialized;
    int terrain_world_size;

    cudaStream_t stream_sky;
    cudaStream_t stream_clouds;
    cudaStream_t stream_terrain;
    cudaStream_t stream_post;

    int frame_width;
    int frame_height;
    int frame_counter;

    void init(int width, int height, int world_size);
    void destroy();

    void beginFrame();

    void renderSky();
    void renderClouds(float time);
    void renderTerrain(const float* d_heightmap, const float* d_vegetation,
                       int world_size);
    void renderCreatures(const void* d_instances, int visible_count);
    void renderHeatmapOverlay(const float* d_scalar, int field_w, int field_h,
                              float min_val, float max_val, int type, float alpha);

    void applyPostProcessing();
    void applyDayNightLighting();

    void updateDayNight(float dt);

    void renderUI(const AnalyticsSnapshot& snapshot, float frame_time,
                  float sim_time, float render_time);

    void endFrame();
    void copyToGLTexture(unsigned int gl_texture);

    Camera getCurrentCamera();
    LightParams getCurrentLight();
};

#endif