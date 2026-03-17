#pragma once

#include "climate_common.cuh"

namespace genesis {
namespace climate {

struct SeasonalState {
    float day_of_year;
    float year_fraction;
    float season_factor;
    float day_length_hours;
    float sun_elevation;
    float vegetation_modifier;
    float temperature_offset;
    int current_tick;
    float ticks_per_day;
    float days_per_year;
};

void initSeasonalState(SeasonalState& state, float ticks_per_day, float days_per_year);

void advanceSeasonalState(SeasonalState& state);

void launchSeasonalTemperatureModKernel(
    float* temperature,
    const float* heightmap,
    int width, int height,
    const SeasonalState& state,
    float axial_tilt,
    cudaStream_t stream);

void launchVegetationSeasonalKernel(
    float* vegetation,
    const float* temperature,
    const float* moisture,
    int width, int height,
    const SeasonalState& state,
    float growth_rate,
    float decay_rate,
    float dt,
    cudaStream_t stream);

void launchDayNightCycleKernel(
    float* solar_flux,
    int width, int height,
    const SeasonalState& state,
    float solar_constant,
    cudaStream_t stream);

}
}