#ifndef CLIMATE_COMMON_CUH
#define CLIMATE_COMMON_CUH

#include <cuda_runtime.h>
#include <math.h>

struct ClimateGridData {
    float* d_temperature;
    float* d_temperature_prev;
    float* d_moisture;
    float* d_moisture_prev;
    float* d_wind_x;
    float* d_wind_y;
    float* d_pressure;
    float* d_cloud_cover;
    float* d_solar_flux;
    float* d_humidity;
    int    world_size;
    int    total_cells;
};

typedef ClimateGridData ClimateGrids;

struct ClimateParams {
    float solar_constant;
    float axial_tilt;
    float heat_diffusion_rate;
    float thermal_conductivity;
    float wind_strength;
    float wind_damping;
    float coriolis_factor;
    float coriolis_strength;
    float evaporation_rate;
    float precipitation_threshold;
    float condensation_threshold;
    float cloud_formation_rate;
    float rainfall_rate;
    float snow_temperature;
    float humidity_diffusion_rate;
    float base_pressure;
    float albedo_land;
    float albedo_water;
    float albedo_ice;
    float greenhouse_factor;
    float dt;
};

#ifndef CUDART_PI_F
#define CUDART_PI_F 3.14159265358979323846f
#endif

struct SeasonState {
    float current_day;
    float year_length;
    float season_angle;
    int   current_season;
    float season_progress;
};

#endif