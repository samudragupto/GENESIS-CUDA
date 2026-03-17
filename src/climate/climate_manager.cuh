#ifndef CLIMATE_MANAGER_CUH
#define CLIMATE_MANAGER_CUH

#include <cuda_runtime.h>
#include "climate_common.cuh"
#include "solar_model.cuh"
#include "heat_diffusion.cuh"
#include "wind_simulation.cuh"
#include "humidity_transport.cuh"
#include "precipitation.cuh"
#include "seasonal_cycle.cuh"

class ClimateManager {
public:
    ClimateGridData grids;
    ClimateParams params;
    SeasonState season;

    cudaStream_t stream_solar;
    cudaStream_t stream_heat;
    cudaStream_t stream_wind;
    cudaStream_t stream_water;

    int world_size;

    void init(int ws);
    void destroy();

    void initFromTerrain(
        const float* d_heightmap,
        int ws,
        float base_temperature,
        float temperature_variation
    );

    void update(
        float dt,
        const float* d_heightmap,
        int ws,
        int tick
    );

    float* getTemperaturePtr();
    float* getMoisturePtr();
    float* getWindXPtr();
    float* getWindYPtr();
    float* getHumidityPtr();
};

#endif