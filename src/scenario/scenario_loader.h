#ifndef SCENARIO_LOADER_H
#define SCENARIO_LOADER_H

struct ScenarioConfig {
    int world_size;
    int max_creatures;
    int initial_creatures;
    int initial_species;

    float water_level;
    float mountain_height;
    int terrain_octaves;
    float terrain_lacunarity;
    float terrain_persistence;
    unsigned int terrain_seed;

    float initial_temperature;
    float temperature_variation;
    float day_length;
    float season_length;

    float vegetation_growth_rate;
    float vegetation_spread_rate;
    float initial_vegetation_density;

    float mutation_rate;
    float crossover_rate;
    float speciation_threshold;

    float base_energy_drain;
    float eat_energy_gain;
    float attack_damage;
    float reproduce_energy_cost;

    int enable_disease;
    float disease_probability;
    int disease_start_tick;

    int render_width;
    int render_height;
    float simulation_speed;
    int auto_save_interval;

    char scenario_name[128];
    char output_directory[256];
};

class ScenarioLoader {
public:
    ScenarioConfig config;

    void loadDefaults();
    void loadIslandScenario();
    void loadContinentScenario();
    void loadOceanWorldScenario();
    void loadDesertScenario();
    void loadIceAgeScenario();
    void loadPandemicScenario();
    void loadMassExtinctionScenario();
    void loadExplosiveRadiationScenario();

    bool loadFromFile(const char* filename);
    bool saveToFile(const char* filename) const;

    void applyCommandLineArgs(int argc, char** argv);

    void printConfig() const;
    ScenarioConfig getConfig() const;
};

#endif