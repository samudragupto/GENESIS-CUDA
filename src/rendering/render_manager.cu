#include "render_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"

void RenderManager::init(int width, int height, int world_size) {
    frame_width = width;
    frame_height = height;
    frame_counter = 0;
    terrain_initialized = 0;
    terrain_world_size = world_size;

    allocateFrameBuffer(framebuffer, width, height);
    allocatePostProcessBuffers(post_buffers, width, height);
    allocateHeatmapData(heatmap, width, height);

    post_params.bloom_threshold = 0.8f;
    post_params.bloom_intensity = 0.3f;
    post_params.bloom_blur_passes = 3;
    post_params.exposure = 1.2f;
    post_params.gamma = 2.2f;
    post_params.saturation = 1.1f;
    post_params.vignette_strength = 0.5f;
    post_params.vignette_radius = 0.7f;
    post_params.fxaa_enabled = 1;

    atmo_params.planet_radius = 6371000.0f;
    atmo_params.atmosphere_radius = 6471000.0f;
    atmo_params.rayleigh_scale_height = 8500.0f;
    atmo_params.mie_scale_height = 1200.0f;
    atmo_params.mie_g = 0.76f;
    atmo_params.rayleigh_coeff = make_float3(5.5e-6f, 13.0e-6f, 22.4e-6f);
    atmo_params.mie_coeff = 21e-6f;
    atmo_params.num_samples = 16;
    atmo_params.num_light_samples = 8;

    cloud_params.cloud_base_height = 200.0f;
    cloud_params.cloud_top_height = 400.0f;
    cloud_params.cloud_coverage = 0.5f;
    cloud_params.cloud_density = 0.3f;
    cloud_params.cloud_speed = 1.0f;
    cloud_params.cloud_scale = 0.002f;
    cloud_params.time = 0.0f;
    cloud_params.num_steps = 32;
    cloud_params.num_light_steps = 6;
    cloud_params.light_absorption = 1.0f;
    cloud_params.ambient_light = 0.2f;

    light.sun_dir_x = 0.5f;
    light.sun_dir_y = 0.7f;
    light.sun_dir_z = 0.3f;
    light.sun_color_r = 1.0f;
    light.sun_color_g = 0.95f;
    light.sun_color_b = 0.9f;
    light.sun_intensity = 1.0f;
    light.ambient_r = 0.15f;
    light.ambient_g = 0.15f;
    light.ambient_b = 0.2f;
    light.ambient_intensity = 0.3f;
    light.time_of_day = 0.25f;

    day_night.time_of_day = 0.25f;
    day_night.day_length = 2400.0f;
    day_night.sun_angle = 0.0f;
    day_night.dawn_start = 0.2f;
    day_night.dawn_end = 0.3f;
    day_night.dusk_start = 0.7f;
    day_night.dusk_end = 0.8f;

    terrain_config.texel_scale = 1.0f;
    terrain_config.height_scale = 100.0f;
    terrain_config.wireframe = 0;
    terrain_config.lod_enabled = 1;
    terrain_config.lod_distance_0 = 100.0f;
    terrain_config.lod_distance_1 = 300.0f;
    terrain_config.lod_distance_2 = 600.0f;
    terrain_config.water_level = WATER_LEVEL;

    terrain_mesh.d_vertices = nullptr;
    terrain_mesh.d_indices = nullptr;
    terrain_mesh.num_vertices = 0;
    terrain_mesh.num_indices = 0;
    terrain_mesh.world_size = world_size;
    terrain_mesh.lod_levels = 3;

    camera_ctrl.init(
        (float)world_size * 0.5f,
        100.0f,
        (float)world_size * 0.5f,
        width, height
    );
    camera_ctrl.lookAt(
        (float)world_size * 0.5f,
        0.0f,
        (float)world_size * 0.5f
    );

    dashboard.init();

    CUDA_CHECK(cudaStreamCreate(&stream_sky));
    CUDA_CHECK(cudaStreamCreate(&stream_clouds));
    CUDA_CHECK(cudaStreamCreate(&stream_terrain));
    CUDA_CHECK(cudaStreamCreate(&stream_post));
}

void RenderManager::destroy() {
    freeFrameBuffer(framebuffer);
    freePostProcessBuffers(post_buffers);
    freeHeatmapData(heatmap);
    dashboard.destroy();

    if (terrain_initialized) {
        freeTerrainMesh(terrain_mesh);
        terrain_initialized = 0;
    }

    cudaStreamDestroy(stream_sky);
    cudaStreamDestroy(stream_clouds);
    cudaStreamDestroy(stream_terrain);
    cudaStreamDestroy(stream_post);
}

void RenderManager::beginFrame() {
    float4 clear = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    clearFrameBuffer(framebuffer, clear, 0);
    frame_counter++;
}

void RenderManager::renderSky() {
    launchSkyDome(framebuffer, camera_ctrl.camera, light,
                  day_night.time_of_day, stream_sky);

    float night_factor = fmaxf(1.0f - light.sun_intensity * 2.0f, 0.0f);
    if (night_factor > 0.1f) {
        launchStarField(framebuffer, camera_ctrl.camera, night_factor,
                       frame_counter, stream_sky);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_sky));
}

void RenderManager::renderClouds(float time) {
    cloud_params.time = time;
    launchCloudRaymarching(framebuffer, camera_ctrl.camera, light,
                           cloud_params, stream_clouds);
    CUDA_CHECK(cudaStreamSynchronize(stream_clouds));
}

void RenderManager::renderTerrain(const float* d_heightmap, const float* d_vegetation,
                                   int world_size) {
    if (!d_heightmap) return;

    if (!terrain_initialized) {
        allocateTerrainMesh(terrain_mesh, world_size, 3);

        launchBuildTerrainVertices(
            terrain_mesh,
            d_heightmap,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            world_size,
            1,
            stream_terrain
        );

        launchBuildTerrainIndices(
            terrain_mesh,
            world_size,
            1,
            stream_terrain
        );

        CUDA_CHECK(cudaStreamSynchronize(stream_terrain));
        terrain_initialized = 1;
        terrain_world_size = world_size;
    }

    if (d_vegetation) {
        launchTerrainColorFromVegetation(
            terrain_mesh,
            d_vegetation,
            d_heightmap,
            world_size,
            terrain_config.water_level,
            stream_terrain
        );
    }

    if (terrain_config.lod_enabled && frame_counter % 30 == 0) {
        int tile_size = 64;
        int tiles_per_side = world_size / tile_size;
        int num_tiles = tiles_per_side * tiles_per_side;

        int* d_lod_per_tile;
        CUDA_CHECK(cudaMalloc(&d_lod_per_tile, num_tiles * sizeof(int)));

        launchTerrainLODSelection(
            terrain_mesh,
            camera_ctrl.camera.pos_x,
            camera_ctrl.camera.pos_y,
            camera_ctrl.camera.pos_z,
            terrain_config,
            world_size,
            d_lod_per_tile,
            tile_size,
            stream_terrain
        );

        CUDA_CHECK(cudaStreamSynchronize(stream_terrain));
        cudaFree(d_lod_per_tile);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_terrain));
}

void RenderManager::renderCreatures(const void* d_instances, int visible_count) {
}

void RenderManager::renderHeatmapOverlay(const float* d_scalar, int field_w, int field_h,
                                          float min_val, float max_val, int type, float alpha) {
    launchScalarToHeatmap(heatmap, d_scalar, field_w, field_h,
                          min_val, max_val, (HeatmapType)type, 0);
    launchGaussianBlurHeatmap(heatmap, 2, 1.5f, 0);
    launchOverlayHeatmap(framebuffer.d_color, heatmap.d_color_output,
                         frame_width, frame_height, alpha, 0);
}

void RenderManager::applyPostProcessing() {
    launchFullPostProcess(framebuffer, post_buffers, post_params, stream_post);
    CUDA_CHECK(cudaStreamSynchronize(stream_post));
}

void RenderManager::applyDayNightLighting() {
    launchApplyDayNightLighting(framebuffer, light, day_night, 0);
}

void RenderManager::updateDayNight(float dt) {
    updateDayNightCycle(day_night, light, dt);
}

void RenderManager::renderUI(const AnalyticsSnapshot& snapshot, float frame_time,
                              float sim_time, float render_time) {
    dashboard.beginFrame();
    dashboard.renderStatsWindow(snapshot);
    dashboard.renderPopulationGraph(snapshot);
    dashboard.renderPerformanceWindow(frame_time, sim_time, render_time);
    dashboard.endFrame();
}

void RenderManager::endFrame() {
}

__global__ void copyToTextureKernel(
    const float4* __restrict__ src,
    unsigned char* __restrict__ dst,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int src_idx = y * width + x;
    int dst_idx = ((height - 1 - y) * width + x) * 4;

    float4 c = src[src_idx];
    dst[dst_idx + 0] = (unsigned char)(fminf(fmaxf(c.x, 0.0f), 1.0f) * 255.0f);
    dst[dst_idx + 1] = (unsigned char)(fminf(fmaxf(c.y, 0.0f), 1.0f) * 255.0f);
    dst[dst_idx + 2] = (unsigned char)(fminf(fmaxf(c.z, 0.0f), 1.0f) * 255.0f);
    dst[dst_idx + 3] = 255;
}

void RenderManager::copyToGLTexture(unsigned int gl_texture) {
}

Camera RenderManager::getCurrentCamera() {
    return camera_ctrl.camera;
}

LightParams RenderManager::getCurrentLight() {
    return light;
}