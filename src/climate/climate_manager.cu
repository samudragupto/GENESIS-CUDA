#include "climate_manager.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/constants.cuh"
#include <cstdio>
#include <cmath>

__global__ void initTemperatureFromTerrainKernel(
    float* __restrict__ temperature,
    float* __restrict__ moisture,
    float* __restrict__ humidity,
    const float* __restrict__ heightmap,
    int world_size,
    float base_temp,
    float temp_variation
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= world_size || y >= world_size) return;

    int idx = y * world_size + x;
    float h = heightmap[idx];
    float lat = (float)y / (float)world_size;
    float lat_factor = 1.0f - fabsf(lat - 0.5f) * 2.0f;

    temperature[idx] = base_temp + lat_factor * temp_variation - h * 20.0f;
    moisture[idx] = (h < WATER_LEVEL) ? 1.0f : fmaxf(0.3f - (h - WATER_LEVEL) * 0.5f, 0.05f);
    humidity[idx] = moisture[idx] * 0.5f;
}

void ClimateManager::init(int ws) {
    world_size = ws;
    int cells = ws * ws;

    CUDA_CHECK(cudaMalloc(&grids.d_temperature, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_temperature_prev, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_moisture, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_moisture_prev, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_wind_x, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_wind_y, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_pressure, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_cloud_cover, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_solar_flux, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&grids.d_humidity, cells * sizeof(float)));
    grids.world_size = ws;
    grids.total_cells = cells;

    CUDA_CHECK(cudaMemset(grids.d_temperature, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_temperature_prev, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_moisture, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_moisture_prev, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_wind_x, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_wind_y, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_pressure, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_cloud_cover, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_solar_flux, 0, cells * sizeof(float)));
    CUDA_CHECK(cudaMemset(grids.d_humidity, 0, cells * sizeof(float)));

    params.solar_constant = 1361.0f;
    params.axial_tilt = 23.44f;
    params.heat_diffusion_rate = 0.1f;
    params.thermal_conductivity = 0.1f;
    params.wind_strength = 1.0f;
    params.wind_damping = 0.98f;
    params.coriolis_factor = 0.01f;
    params.coriolis_strength = 0.01f;
    params.evaporation_rate = 0.001f;
    params.precipitation_threshold = 0.7f;
    params.condensation_threshold = 0.8f;
    params.cloud_formation_rate = 0.01f;
    params.rainfall_rate = 0.005f;
    params.snow_temperature = 0.0f;
    params.humidity_diffusion_rate = 0.05f;
    params.base_pressure = 101325.0f;
    params.albedo_land = 0.3f;
    params.albedo_water = 0.06f;
    params.albedo_ice = 0.8f;
    params.greenhouse_factor = 1.0f;
    params.dt = 1.0f;

    season.current_day = 0.0f;
    season.year_length = 365.0f;
    season.season_angle = 0.0f;
    season.current_season = 0;
    season.season_progress = 0.0f;

    CUDA_CHECK(cudaStreamCreate(&stream_solar));
    CUDA_CHECK(cudaStreamCreate(&stream_heat));
    CUDA_CHECK(cudaStreamCreate(&stream_wind));
    CUDA_CHECK(cudaStreamCreate(&stream_water));
}

void ClimateManager::destroy() {
    cudaFree(grids.d_temperature);
    cudaFree(grids.d_temperature_prev);
    cudaFree(grids.d_moisture);
    cudaFree(grids.d_moisture_prev);
    cudaFree(grids.d_wind_x);
    cudaFree(grids.d_wind_y);
    cudaFree(grids.d_pressure);
    cudaFree(grids.d_cloud_cover);
    cudaFree(grids.d_solar_flux);
    cudaFree(grids.d_humidity);

    cudaStreamDestroy(stream_solar);
    cudaStreamDestroy(stream_heat);
    cudaStreamDestroy(stream_wind);
    cudaStreamDestroy(stream_water);
}

void ClimateManager::initFromTerrain(
    const float* d_heightmap, int ws,
    float base_temperature, float temperature_variation
) {
    dim3 block(16, 16);
    dim3 grid((ws + 15) / 16, (ws + 15) / 16);
    initTemperatureFromTerrainKernel<<<grid, block>>>(
        grids.d_temperature, grids.d_moisture, grids.d_humidity,
        d_heightmap, ws, base_temperature, temperature_variation
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(grids.d_temperature_prev, grids.d_temperature,
        (size_t)ws * ws * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(grids.d_moisture_prev, grids.d_moisture,
        (size_t)ws * ws * sizeof(float), cudaMemcpyDeviceToDevice));
}

void ClimateManager::update(
    float dt, const float* d_heightmap, int ws, int tick
) {
    params.dt = dt;

    float sun_angle = sinf((float)tick * 0.01f) * 0.5f + 0.5f;

    launchSolarRadiationKernel(
        grids.d_solar_flux, d_heightmap, ws,
        params.solar_constant, sun_angle,
        params.axial_tilt, params.dt,
        stream_solar
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_solar));

    launchHeatDiffusionKernel(
        grids.d_temperature, grids.d_solar_flux,
        d_heightmap, ws,
        params.thermal_conductivity, params.dt,
        stream_heat
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_heat));

    launchPressureComputeKernel(
        grids.d_pressure, grids.d_temperature, ws,
        params.base_pressure, stream_wind
    );

    launchWindUpdateKernel(
        grids.d_wind_x, grids.d_wind_y,
        grids.d_pressure, ws,
        params.wind_damping, params.coriolis_factor,
        params.dt, stream_wind
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_wind));

    launchEvaporationKernel(
        grids.d_humidity, grids.d_temperature,
        grids.d_moisture, d_heightmap, ws,
        params.evaporation_rate, WATER_LEVEL,
        params.dt, stream_water
    );

    launchHumidityAdvectionKernel(
        grids.d_humidity, grids.d_wind_x, grids.d_wind_y,
        ws, params.dt, stream_water
    );
    CUDA_CHECK(cudaStreamSynchronize(stream_water));

    if (tick % 10 == 0) {
        launchCloudFormationKernel(
            grids.d_cloud_cover, grids.d_humidity,
            grids.d_temperature, ws, params, stream_water
        );

        launchRainfallKernel(
            grids.d_moisture, grids.d_cloud_cover,
            grids.d_humidity, grids.d_temperature,
            ws, params, stream_water
        );
        CUDA_CHECK(cudaStreamSynchronize(stream_water));
    }

    season.current_day += params.dt / 24.0f;
    if (season.current_day > season.year_length)
        season.current_day -= season.year_length;
    season.season_angle = season.current_day / season.year_length * 2.0f * 3.14159265f;
    season.current_season = (int)(season.current_day / season.year_length * 4.0f) % 4;
}

float* ClimateManager::getTemperaturePtr() { return grids.d_temperature; }
float* ClimateManager::getMoisturePtr() { return grids.d_moisture; }
float* ClimateManager::getWindXPtr() { return grids.d_wind_x; }
float* ClimateManager::getWindYPtr() { return grids.d_wind_y; }
float* ClimateManager::getHumidityPtr() { return grids.d_humidity; }