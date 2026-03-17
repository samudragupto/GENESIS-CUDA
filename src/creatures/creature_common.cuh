#ifndef CREATURE_COMMON_CUH
#define CREATURE_COMMON_CUH

#include <cuda_runtime.h>
#include "../neural/neural_common.cuh"

enum CreatureState : int {
    STATE_FORAGING = 0,
    STATE_EATING = 1,
    STATE_HUNTING = 2,
    STATE_FLEEING = 3,
    STATE_SOCIALIZING = 4,
    STATE_MATING = 5,
    STATE_COMBAT = 6,
    STATE_DEAD = 7
};

struct CreatureData {
    float* d_pos_x;
    float* d_pos_y;
    float* d_vel_x;
    float* d_vel_y;
    float* d_energy;
    float* d_health;
    int*   d_age;
    int*   d_species_id;
    int*   d_state;
    float* d_repro_cooldown;
    float* d_genomes;
    float* d_neural_weights;
    int*   d_alive;
    int    max_creatures;
    int*   d_num_alive;
};

struct InteractionBuffers {
    int*   d_reproduce_flag;
    int*   d_mate_index;
    int*   d_mating_lock;
    int*   d_spawn_counter;
    int*   d_new_creature_indices;
    int    max_creatures;
};

struct RenderInstance {
    float x, y, z;
    float r, g, b;
    float scale;
    float rotation;
};

void allocateCreatureData(CreatureData& data, int max_creatures);
void freeCreatureData(CreatureData& data);
void allocateInteractionBuffers(InteractionBuffers& buf, int max_creatures);
void freeInteractionBuffers(InteractionBuffers& buf);

#endif