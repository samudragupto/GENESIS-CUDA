#include "scenario_loader.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

void ScenarioLoader::loadDefaults() {
    config.world_size = 1024;
    config.max_creatures = 500000;
    config.initial_creatures = 10000;
    config.initial_species = 10;
    config.water_level = 0.35f;
    config.mountain_height = 1.0f;
    config.terrain_octaves = 8;
    config.terrain_lacunarity = 2.0f;
    config.terrain_persistence = 0.5f;
    config.terrain_seed = 42;
    config.initial_temperature = 20.0f;
    config.temperature_variation = 15.0f;
    config.day_length = 2400.0f;
    config.season_length = 24000.0f;
    config.vegetation_growth_rate = 0.01f;
    config.vegetation_spread_rate = 0.005f;
    config.initial_vegetation_density = 0.5f;
    config.mutation_rate = 0.02f;
    config.crossover_rate = 0.7f;
    config.speciation_threshold = 0.3f;
    config.base_energy_drain = 0.005f;
    config.eat_energy_gain = 0.5f;
    config.attack_damage = 0.3f;
    config.reproduce_energy_cost = 0.2f;
    config.enable_disease = 0;
    config.disease_probability = 0.001f;
    config.disease_start_tick = 5000;
    config.render_width = 1920;
    config.render_height = 1080;
    config.simulation_speed = 1.0f;
    config.auto_save_interval = 5000;
    strcpy(config.scenario_name, "Default");
    strcpy(config.output_directory, "./output");
}

void ScenarioLoader::loadIslandScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Island Archipelago");
    config.world_size = 512;
    config.water_level = 0.55f;
    config.initial_creatures = 5000;
    config.max_creatures = 200000;
    config.terrain_seed = 12345;
    config.initial_temperature = 25.0f;
    config.vegetation_growth_rate = 0.015f;
}

void ScenarioLoader::loadContinentScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Continental");
    config.world_size = 2048;
    config.water_level = 0.3f;
    config.initial_creatures = 50000;
    config.max_creatures = 1000000;
    config.terrain_seed = 67890;
    config.temperature_variation = 25.0f;
}

void ScenarioLoader::loadOceanWorldScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Ocean World");
    config.world_size = 1024;
    config.water_level = 0.7f;
    config.initial_creatures = 3000;
    config.max_creatures = 100000;
    config.terrain_seed = 11111;
    config.initial_temperature = 22.0f;
}

void ScenarioLoader::loadDesertScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Desert World");
    config.world_size = 1024;
    config.water_level = 0.2f;
    config.initial_creatures = 5000;
    config.initial_temperature = 35.0f;
    config.temperature_variation = 20.0f;
    config.vegetation_growth_rate = 0.003f;
    config.vegetation_spread_rate = 0.001f;
    config.initial_vegetation_density = 0.1f;
    config.terrain_seed = 22222;
}

void ScenarioLoader::loadIceAgeScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Ice Age");
    config.world_size = 1024;
    config.water_level = 0.25f;
    config.initial_creatures = 8000;
    config.initial_temperature = 5.0f;
    config.temperature_variation = 10.0f;
    config.vegetation_growth_rate = 0.005f;
    config.base_energy_drain = 0.008f;
    config.terrain_seed = 33333;
}

void ScenarioLoader::loadPandemicScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Pandemic");
    config.initial_creatures = 30000;
    config.enable_disease = 1;
    config.disease_probability = 0.01f;
    config.disease_start_tick = 1000;
    config.terrain_seed = 44444;
}

void ScenarioLoader::loadMassExtinctionScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Mass Extinction");
    config.initial_creatures = 50000;
    config.initial_species = 50;
    config.base_energy_drain = 0.01f;
    config.vegetation_growth_rate = 0.002f;
    config.initial_temperature = 40.0f;
    config.temperature_variation = 30.0f;
    config.terrain_seed = 55555;
}

void ScenarioLoader::loadExplosiveRadiationScenario() {
    loadDefaults();
    strcpy(config.scenario_name, "Explosive Radiation");
    config.initial_creatures = 1000;
    config.initial_species = 2;
    config.mutation_rate = 0.05f;
    config.speciation_threshold = 0.15f;
    config.vegetation_growth_rate = 0.02f;
    config.eat_energy_gain = 0.8f;
    config.base_energy_drain = 0.003f;
    config.terrain_seed = 66666;
}

bool ScenarioLoader::loadFromFile(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) return false;

    char line[512];
    char key[128];
    char value[256];

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        if (sscanf(line, "%127[^=]=%255s", key, value) != 2) continue;

        if (strcmp(key, "world_size") == 0) config.world_size = atoi(value);
        else if (strcmp(key, "max_creatures") == 0) config.max_creatures = atoi(value);
        else if (strcmp(key, "initial_creatures") == 0) config.initial_creatures = atoi(value);
        else if (strcmp(key, "water_level") == 0) config.water_level = (float)atof(value);
        else if (strcmp(key, "terrain_seed") == 0) config.terrain_seed = (unsigned int)atoi(value);
        else if (strcmp(key, "temperature") == 0) config.initial_temperature = (float)atof(value);
        else if (strcmp(key, "mutation_rate") == 0) config.mutation_rate = (float)atof(value);
        else if (strcmp(key, "speciation_threshold") == 0) config.speciation_threshold = (float)atof(value);
        else if (strcmp(key, "scenario_name") == 0) strncpy(config.scenario_name, value, 127);
        else if (strcmp(key, "output_directory") == 0) strncpy(config.output_directory, value, 255);
        else if (strcmp(key, "enable_disease") == 0) config.enable_disease = atoi(value);
        else if (strcmp(key, "simulation_speed") == 0) config.simulation_speed = (float)atof(value);
        else if (strcmp(key, "render_width") == 0) config.render_width = atoi(value);
        else if (strcmp(key, "render_height") == 0) config.render_height = atoi(value);
    }

    fclose(fp);
    return true;
}

bool ScenarioLoader::saveToFile(const char* filename) const {
    FILE* fp = fopen(filename, "w");
    if (!fp) return false;

    fprintf(fp, "# GENESIS Scenario Configuration\n");
    fprintf(fp, "scenario_name=%s\n", config.scenario_name);
    fprintf(fp, "world_size=%d\n", config.world_size);
    fprintf(fp, "max_creatures=%d\n", config.max_creatures);
    fprintf(fp, "initial_creatures=%d\n", config.initial_creatures);
    fprintf(fp, "initial_species=%d\n", config.initial_species);
    fprintf(fp, "water_level=%.3f\n", config.water_level);
    fprintf(fp, "mountain_height=%.3f\n", config.mountain_height);
    fprintf(fp, "terrain_seed=%u\n", config.terrain_seed);
    fprintf(fp, "terrain_octaves=%d\n", config.terrain_octaves);
    fprintf(fp, "temperature=%.1f\n", config.initial_temperature);
    fprintf(fp, "temperature_variation=%.1f\n", config.temperature_variation);
    fprintf(fp, "day_length=%.1f\n", config.day_length);
    fprintf(fp, "vegetation_growth_rate=%.4f\n", config.vegetation_growth_rate);
    fprintf(fp, "vegetation_spread_rate=%.4f\n", config.vegetation_spread_rate);
    fprintf(fp, "mutation_rate=%.4f\n", config.mutation_rate);
    fprintf(fp, "crossover_rate=%.4f\n", config.crossover_rate);
    fprintf(fp, "speciation_threshold=%.4f\n", config.speciation_threshold);
    fprintf(fp, "base_energy_drain=%.4f\n", config.base_energy_drain);
    fprintf(fp, "eat_energy_gain=%.4f\n", config.eat_energy_gain);
    fprintf(fp, "enable_disease=%d\n", config.enable_disease);
    fprintf(fp, "disease_start_tick=%d\n", config.disease_start_tick);
    fprintf(fp, "render_width=%d\n", config.render_width);
    fprintf(fp, "render_height=%d\n", config.render_height);
    fprintf(fp, "simulation_speed=%.2f\n", config.simulation_speed);
    fprintf(fp, "auto_save_interval=%d\n", config.auto_save_interval);
    fprintf(fp, "output_directory=%s\n", config.output_directory);

    fclose(fp);
    return true;
}

void ScenarioLoader::applyCommandLineArgs(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--scenario") == 0 && i + 1 < argc) {
            if (strcmp(argv[i+1], "island") == 0) loadIslandScenario();
            else if (strcmp(argv[i+1], "continent") == 0) loadContinentScenario();
            else if (strcmp(argv[i+1], "ocean") == 0) loadOceanWorldScenario();
            else if (strcmp(argv[i+1], "desert") == 0) loadDesertScenario();
            else if (strcmp(argv[i+1], "iceage") == 0) loadIceAgeScenario();
            else if (strcmp(argv[i+1], "pandemic") == 0) loadPandemicScenario();
            else if (strcmp(argv[i+1], "extinction") == 0) loadMassExtinctionScenario();
            else if (strcmp(argv[i+1], "radiation") == 0) loadExplosiveRadiationScenario();
            i++;
        }
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            loadFromFile(argv[++i]);
        }
        else if (strcmp(argv[i], "--world-size") == 0 && i + 1 < argc) {
            config.world_size = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--creatures") == 0 && i + 1 < argc) {
            config.initial_creatures = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--max-creatures") == 0 && i + 1 < argc) {
            config.max_creatures = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            config.terrain_seed = (unsigned int)atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            strncpy(config.output_directory, argv[++i], 255);
        }
    }
}

void ScenarioLoader::printConfig() const {
    printf("\n============ GENESIS Configuration ============\n");
    printf("Scenario: %s\n", config.scenario_name);
    printf("World: %dx%d | Water Level: %.2f\n",
           config.world_size, config.world_size, config.water_level);
    printf("Creatures: %d initial | %d max | %d species\n",
           config.initial_creatures, config.max_creatures, config.initial_species);
    printf("Terrain: seed=%u octaves=%d\n",
           config.terrain_seed, config.terrain_octaves);
    printf("Climate: temp=%.1f var=%.1f day=%.0f\n",
           config.initial_temperature, config.temperature_variation, config.day_length);
    printf("Genetics: mutation=%.3f crossover=%.3f speciation=%.3f\n",
           config.mutation_rate, config.crossover_rate, config.speciation_threshold);
    printf("Disease: %s (start tick %d)\n",
           config.enable_disease ? "enabled" : "disabled", config.disease_start_tick);
    printf("Render: %dx%d | Speed: %.1fx\n",
           config.render_width, config.render_height, config.simulation_speed);
    printf("Output: %s | AutoSave: every %d ticks\n",
           config.output_directory, config.auto_save_interval);
    printf("================================================\n\n");
}

ScenarioConfig ScenarioLoader::getConfig() const {
    return config;
}