#include "seasonal_cycle.cuh"
#include <cmath>

namespace genesis {
namespace climate {

void initSeasonalState(SeasonalState& state, float ticks_per_day, float days_per_year) {
    state.day_of_year = 0.0f;
    state.year_fraction = 0.0f;
    state.season_factor = 0.0f;
    state.day_length_hours = 12.0f;
    state.sun_elevation = 0.5f;
    state.vegetation_modifier = 1.0f;
    state.temperature_offset = 0.0f;
    state.current_tick = 0;
    state.ticks_per_day = ticks_per_day;
    state.days_per_year = days_per_year;
}

void advanceSeasonalState(SeasonalState& state) {
    state.current_tick++;
    state.day_of_year = fmodf(
        (float)state.current_tick / state.ticks_per_day,
        state.days_per_year);
    state.year_fraction = state.day_of_year / state.days_per_year;

    state.season_factor = sinf(state.year_fraction * 2.0f * 3.14159265f);

    state.day_length_hours = 12.0f + 4.0f * state.season_factor;

    float time_of_day = fmodf(
        (float)state.current_tick / state.ticks_per_day, 1.0f);
    float dawn = (12.0f - state.day_length_hours * 0.5f) / 24.0f;
    float dusk = (12.0f + state.day_length_hours * 0.5f) / 24.0f;

    if (time_of_day >= dawn && time_of_day <= dusk) {
        float day_progress = (time_of_day - dawn) / (dusk - dawn);
        state.sun_elevation = sinf(day_progress * 3.14159265f);
    } else {
        state.sun_elevation = 0.0f;
    }

    state.temperature_offset = state.season_factor * 15.0f;

    float spring_factor = fmaxf(0.0f,
        cosf((state.year_fraction - 0.25f) * 2.0f * 3.14159265f));
    state.vegetation_modifier = 0.3f + 0.7f * spring_factor;
}

__global__ void seasonalTemperatureModKernel(
    float* __restrict__ temperature,
    const float* __restrict__ heightmap,
    int width,
    int height,
    float season_factor,
    float axial_tilt,
    float temperature_offset)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float latitude = ((float)y / (float)height - 0.5f) * 2.0f;
    float seasonal_effect = axial_tilt * season_factor * latitude;

    float altitude = heightmap[idx];
    float lapse_rate = -6.5f * altitude;

    temperature[idx] += seasonal_effect + lapse_rate * 0.001f + temperature_offset * 0.01f;
    temperature[idx] = fmaxf(200.0f, fminf(340.0f, temperature[idx]));
}

__global__ void vegetationSeasonalKernel(
    float* __restrict__ vegetation,
    const float* __restrict__ temperature,
    const float* __restrict__ moisture,
    int width,
    int height,
    float veg_modifier,
    float growth_rate,
    float decay_rate,
    float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float temp = temperature[idx];
    float moist = moisture[idx];
    float veg = vegetation[idx];

    float temp_celsius = temp - 273.15f;
    float temp_suitability;
    if (temp_celsius < 0.0f) {
        temp_suitability = 0.0f;
    } else if (temp_celsius < 10.0f) {
        temp_suitability = temp_celsius / 10.0f;
    } else if (temp_celsius < 30.0f) {
        temp_suitability = 1.0f;
    } else if (temp_celsius < 45.0f) {
        temp_suitability = 1.0f - (temp_celsius - 30.0f) / 15.0f;
    } else {
        temp_suitability = 0.0f;
    }

    float moisture_suitability = fminf(1.0f, moist * 2.0f);

    float growth_potential = temp_suitability * moisture_suitability * veg_modifier;

    float growth = growth_rate * growth_potential * (1.0f - veg) * dt;
    float decay = decay_rate * (1.0f - growth_potential) * veg * dt;

    veg += growth - decay;
    vegetation[idx] = fmaxf(0.0f, fminf(1.0f, veg));
}

__global__ void dayNightCycleKernel(
    float* __restrict__ solar_flux,
    int width,
    int height,
    float sun_elevation,
    float solar_constant,
    float season_factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    float latitude = ((float)y / (float)height - 0.5f) * 3.14159265f;
    float lat_factor = cosf(latitude);

    float tilt_offset = 0.4f * season_factor * sinf(latitude);
    float effective_elevation = fmaxf(0.0f, sun_elevation + tilt_offset);

    solar_flux[idx] = solar_constant * effective_elevation * lat_factor;
}

void launchSeasonalTemperatureModKernel(
    float* temperature,
    const float* heightmap,
    int width, int height,
    const SeasonalState& state,
    float axial_tilt,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    seasonalTemperatureModKernel<<<grid, block, 0, stream>>>(
        temperature, heightmap,
        width, height,
        state.season_factor,
        axial_tilt,
        state.temperature_offset);
}

void launchVegetationSeasonalKernel(
    float* vegetation,
    const float* temperature,
    const float* moisture,
    int width, int height,
    const SeasonalState& state,
    float growth_rate,
    float decay_rate,
    float dt,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    vegetationSeasonalKernel<<<grid, block, 0, stream>>>(
        vegetation, temperature, moisture,
        width, height,
        state.vegetation_modifier,
        growth_rate, decay_rate, dt);
}

void launchDayNightCycleKernel(
    float* solar_flux,
    int width, int height,
    const SeasonalState& state,
    float solar_constant,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    dayNightCycleKernel<<<grid, block, 0, stream>>>(
        solar_flux, width, height,
        state.sun_elevation,
        solar_constant,
        state.season_factor);
}

}
}