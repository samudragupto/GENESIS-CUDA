# GENESIS: GPU-Accelerated Planetary Evolution & Ecosystem Simulator

![CUDA](https://img.shields.io/badge/CUDA-11.8%20%7C%2012.x-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![CMake](https://img.shields.io/badge/CMake-3.18+-064F8C?style=for-the-badge&logo=cmake&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey?style=for-the-badge)

**GENESIS** is a massive-scale artificial life and planetary simulator written entirely in **C++ and CUDA**. By offloading 100% of the computational workload to the GPU, GENESIS eliminates CPU-GPU memory bottlenecks, allowing for the real-time simulation of millions of interacting creatures, dynamic climate patterns, fluid dynamics, and genetic evolution.

## Technical Highlights

* **Custom Neural AI Engine:** Every creature possesses a unique neural network. Feed-forward evaluation is executed in parallel via custom Batched GEMM CUDA kernels, with weights dynamically decoded from organism genomes directly in VRAM.
* **Genetic Algorithms:** Organism traits (speed, vision, diet, lifespan, neural topology) are encoded in a 256-float genome. Crossover, mutation (via `cuRAND`), and distance-based speciation occur entirely on the GPU.
* **Spatial Acceleration:** Utilizes NVIDIA Thrust for parallel radix sorting and spatial hashing, achieving O(1) neighbor lookups and collision detection for massive entity counts.
* **Climate & Cellular Automata:** Procedural Perlin noise heightmaps with thermal/hydraulic erosion. Cellular automata systems simulate 2D heat diffusion, wind advection, moisture transport, and dynamic vegetation growth.
* **cosystem Dynamics:** Multi-trophic food web processing, spatial disease propagation (SIR model), and population-carrying capacity enforced via parallel reductions.

---
## Numerical Stability & Precision
To maintain stability across interacting differential equations (Fluid, Heat, Advection) running asynchronously on the GPU, specific constraints are enforced:
* **Precision Strategy:** The simulation utilizes **FP32 (Single Precision `float`)** universally. While FP16/Tensor Cores are standard for Deep Learning, they were deliberately avoided here. Coordinate tracking across a 4096x4096 planetary grid requires high mantissa resolution to prevent spatial quantization, and the custom per-agent MLP evaluation is primarily memory-bandwidth bound, not compute bound.
* **Time Integration:** Fixed timestep ($\Delta t$) decoupled by system. Global ecosystem updates operate at $\Delta t = 1.0$, while local physics and SPH integrations operate at $\Delta t = 0.001$.
* **Heat Diffusion Solver:** Solved via parallel Jacobi iteration. To ensure convergence and prevent thermal explosion, the diffusion kernel strictly bounds maximum energy transfer per cell and executes fixed 50-100 iterations per macro-tick.
* **Fluid Dynamics (SPH):** Adheres to the **CFL (Courant–Friedrichs–Lewy) condition** via velocity clamping. Tait Equation of State is used for weak compressibility, with density strictly clamped to prevent division-by-zero singularities during neighbor deficits.


## System Architecture
GENESIS utilizes a highly concurrent, multi-stream architecture. The CPU acts only as an orchestrator, dispatching kernels to the GPU.

```mermaid
graph TB

%% =========================
%% HOST CONTROL LAYER
%% =========================
subgraph HOST["HOST CONTROL LAYER (CPU)"]
    CMD[Command & Control Interface]
    CFG[Configuration Manager]
    MON[Performance Monitor]
    DISK[Disk I/O & Checkpointing]
end

%% =========================
%% GPU ORCHESTRATION
%% =========================
subgraph ORCH["GPU ORCHESTRATION LAYER"]
    SCHED[Kernel Scheduler & Stream Manager]
    MEM[Unified Memory Pool Manager]
    SYNC[Multi-GPU Synchronizer]
    EVENT[Event & Dependency Tracker]
end

CMD --> SCHED
CFG --> SCHED
MON --> SCHED

%% =========================
%% GPU SIMULATION CORE
%% =========================
subgraph CORE["GPU SIMULATION CORE"]

    subgraph PHYS["Physics Engine"]
        SPH[SPH Fluid Dynamics]
        RIGID[Rigid Body Physics]
        COLL[Spatial Hash Collision Detection]
        PART[Particle Systems]
    end

    subgraph WORLD["World Engine"]
        TERRAIN[Procedural Terrain Generator]
        CELLAUT[Cellular Automata]
        GEO[Geological Simulation]
        BIOME[Biome Classification]
    end

    subgraph LIFE["Life Engine"]
        GENE[Genetic Algorithms]
        NEURAL[GPU Neural Networks]
        CREATURE[Creature Lifecycle]
        BEHAV[Behavior Simulation]
    end

    subgraph ECO["Ecosystem Engine"]
        FOOD[Food Web Energy Flow]
        POP[Population Dynamics]
        DISEASE[Disease Spread Model]
        RESOURCE[Resource Distribution]
    end

    subgraph CLIMATE["Climate Engine"]
        ATMOS[Atmospheric Simulation]
        WEATHER[Weather System]
        OCEAN[Ocean Currents]
        SOLAR[Solar Radiation]
    end

end

%% Scheduler control
SCHED --> PHYS
SCHED --> WORLD
SCHED --> LIFE
SCHED --> ECO
SCHED --> CLIMATE

MEM --> PHYS
MEM --> WORLD
MEM --> LIFE
MEM --> ECO
MEM --> CLIMATE

%% =========================
%% VISUALIZATION
%% =========================
subgraph VIS["GPU VISUALIZATION PIPELINE"]
    RENDER[CUDA OpenGL Renderer]
    RAY[Ray Marching Engine]
    POST[Post Processing]
    UI[ImGui Interface]
end

PHYS --> RENDER
WORLD --> RENDER
LIFE --> RENDER
CLIMATE --> RENDER

RENDER --> RAY --> POST --> UI

%% =========================
%% ANALYTICS
%% =========================
subgraph ANALYTICS["GPU ANALYTICS ENGINE"]
    STATS[Statistical Kernels]
    PHYLO[Phylogenetic Tree Builder]
    HEATMAP[Spatial Heatmaps]
    EXPORT[Simulation Data Export]
end

ECO --> STATS
STATS --> PHYLO
STATS --> HEATMAP
PHYLO --> EXPORT

DISK --> EXPORT
```
## Simulation Loop Flowchart

```mermaid
flowchart TD
    START([Simulation Tick Begins]) --> STREAM_SETUP[Setup CUDA Streams & Events]

    STREAM_SETUP --> PARALLEL_1

    subgraph PARALLEL_1 [" PARALLEL STREAM GROUP 1 "]
        direction LR
        CLIMATE_TICK[Climate Kernel Launch<br/>Temperature, Wind, Rain]
        TERRAIN_TICK[Terrain Update Kernel<br/>Erosion, Sediment]
        RESOURCE_TICK[Resource Regeneration<br/>Plants, Water, Minerals]
    end

    PARALLEL_1 --> SYNC1[cudaStreamSynchronize - Barrier 1]

    SYNC1 --> PARALLEL_2

    subgraph PARALLEL_2 [" PARALLEL STREAM GROUP 2 "]
        direction LR
        SENSE[Creature Sensory Input Kernel<br/>Gather local environment data]
        FOODWEB[Food Web Energy<br/>Distribution Kernel]
    end

    PARALLEL_2 --> SYNC2[cudaStreamSynchronize - Barrier 2]

    SYNC2 --> NEURAL_FORWARD[Neural Network Forward Pass Kernel<br/>All creatures simultaneously]

    NEURAL_FORWARD --> DECISION[Decision Extraction Kernel<br/>Movement, Eat, Reproduce, Fight]

    DECISION --> PARALLEL_3

    subgraph PARALLEL_3 [" PARALLEL STREAM GROUP 3 "]
        direction LR
        MOVE[Movement & Physics Kernel]
        INTERACT[Interaction Resolution Kernel<br/>Combat, Mating, Feeding]
    end

    PARALLEL_3 --> SYNC3[cudaStreamSynchronize - Barrier 3]

    SYNC3 --> COLLISION[Spatial Hash Rebuild +<br/>Collision Detection Kernel]

    COLLISION --> PARALLEL_4

    subgraph PARALLEL_4 [" PARALLEL STREAM GROUP 4 "]
        direction LR
        LIFECYCLE[Lifecycle Kernel<br/>Age, Energy, Death]
        REPRODUCE[Reproduction Kernel<br/>Crossover, Mutation on GPU]
        DISEASE_K[Disease Spread Kernel]
    end

    PARALLEL_4 --> SYNC4[cudaStreamSynchronize - Barrier 4]

    SYNC4 --> COMPACT[Dead Creature Compaction Kernel<br/>Stream Compaction with Thrust]

    COMPACT --> PARALLEL_5

    subgraph PARALLEL_5 [" PARALLEL STREAM GROUP 5 "]
        direction LR
        STATS_K[Statistics Gathering Kernel<br/>Parallel Reduction]
        SPATIAL[Spatial Grid Update Kernel]
    end

    PARALLEL_5 --> RENDER_PHASE[Render Frame<br/>CUDA-OpenGL Interop]

    RENDER_PHASE --> CHECKPOINT{Every N ticks?}
    CHECKPOINT -->|Yes| SAVE[Async Memcpy to Host +<br/>Checkpoint to Disk]
    CHECKPOINT -->|No| DONE
    SAVE --> DONE([Simulation Tick Complete])
```

## Memory Architecture

```mermaid
graph TB
    subgraph "GPU GLOBAL MEMORY LAYOUT"
        subgraph "WORLD DATA (~2 GB)"
            HEIGHTMAP[Heightmap Buffer<br/>4096x4096 float]
            MOISTURE[Moisture Map<br/>4096x4096 float]
            TEMP_MAP[Temperature Map<br/>4096x4096 float]
            WIND_MAP[Wind Vector Field<br/>4096x4096 float2]
            VEGETATION[Vegetation Density Map<br/>4096x4096 float]
            MINERAL[Mineral Map<br/>4096x4096 float4]
            BIOME_MAP[Biome ID Map<br/>4096x4096 int]
        end

        subgraph "CREATURE SOA DATA (~4 GB for 2M creatures)"
            POS_BUF[Position Buffer<br/>float3 x 2M]
            VEL_BUF[Velocity Buffer<br/>float3 x 2M]
            ENERGY_BUF[Energy Buffer<br/>float x 2M]
            AGE_BUF[Age Buffer<br/>int x 2M]
            GENOME_BUF[Genome Buffer<br/>float x 2M x 256 genes]
            NEURAL_W[Neural Weights Buffer<br/>float x 2M x 1024 weights]
            SPECIES_BUF[Species ID Buffer<br/>int x 2M]
            STATE_BUF[State Buffer<br/>int x 2M]
            HEALTH_BUF[Health Buffer<br/>float x 2M]
            REPRO_BUF[Reproduction Cooldown<br/>float x 2M]
        end

        subgraph "SPATIAL ACCELERATION"
            HASH_TABLE[Spatial Hash Table<br/>int x 4M]
            CELL_START[Cell Start Array<br/>int x 1M cells]
            CELL_END[Cell End Array<br/>int x 1M cells]
            SORTED_IDX[Sorted Index Array<br/>int x 2M]
        end

        subgraph "FLUID SIMULATION"
            FLUID_POS[Fluid Particle Positions<br/>float3 x 500K]
            FLUID_VEL[Fluid Velocity<br/>float3 x 500K]
            FLUID_DENS[Fluid Density<br/>float x 500K]
            FLUID_PRESS[Fluid Pressure<br/>float x 500K]
        end

        subgraph "ANALYTICS BUFFERS"
            POP_HIST[Population Histogram<br/>int x 10K species x 1K ticks]
            GENE_FREQ[Gene Frequency Buffer<br/>float x 256 x 1K ticks]
            PHYLO_TREE[Phylogenetic Tree Nodes<br/>struct x 100K]
        end
    end

    subgraph "SHARED MEMORY USAGE (Per Block)"
        TILE[Neighbor Tile Cache<br/>48KB per block]
        REDUCE[Reduction Scratch<br/>Shared within warps]
    end

    subgraph "TEXTURE MEMORY"
        TERRAIN_TEX[Terrain Textures<br/>Bound as cudaTextureObject]
        LOOKUP[Lookup Tables<br/>Activation functions, etc.]
    end

    subgraph "CONSTANT MEMORY"
        SIM_PARAMS[Simulation Parameters<br/>~4KB]
        PHYSICS_C[Physics Constants<br/>~1KB]
    end
```

## Creature Genome & Neural Architecture

```mermaid
graph LR
    subgraph "GENOME STRUCTURE (256 floats per creature)"
        subgraph "MORPHOLOGY GENES [0-31]"
            G_SIZE[Body Size]
            G_SPEED[Max Speed]
            G_SENSE[Sense Radius]
            G_COLOR[Color RGB]
            G_ARMOR[Armor Thickness]
            G_LIMBS[Limb Count]
        end

        subgraph "METABOLISM GENES [32-63]"
            G_DIET[Diet Type<br/>Herbivore↔Carnivore]
            G_EFFICIENCY[Energy Efficiency]
            G_LIFESPAN[Max Lifespan]
            G_REPRO_RATE[Reproduction Rate]
            G_TEMP_TOL[Temperature Tolerance]
            G_WATER[Water Requirement]
        end

        subgraph "BEHAVIOR GENES [64-95]"
            G_AGGRO[Aggression Level]
            G_SOCIAL[Social Tendency]
            G_CURIOSITY[Curiosity]
            G_FEAR[Fear Response]
            G_TERR[Territorial Radius]
        end

        subgraph "NEURAL TOPOLOGY GENES [96-255]"
            G_WEIGHTS_SEED[Weight Initialization Seeds]
            G_TOPOLOGY[Hidden Layer Sizes]
            G_ACTIVATION[Activation Function Selectors]
            G_LEARNING[Learning Rate Gene]
        end
    end

    subgraph "NEURAL NETWORK PER CREATURE"
        direction TB
        INPUT["INPUT LAYER (24 neurons)<br/>═══════════════<br/>8x Nearest creature distance/type<br/>4x Resource detection<br/>3x Terrain slope<br/>2x Temperature/moisture<br/>1x Own energy level<br/>1x Own health<br/>2x Wind direction<br/>1x Day/night cycle<br/>1x Age fraction<br/>1x Reproductive readiness"]

        HIDDEN1["HIDDEN LAYER 1<br/>32 neurons<br/>Activation from genome"]

        HIDDEN2["HIDDEN LAYER 2<br/>16 neurons<br/>Activation from genome"]

        OUTPUT["OUTPUT LAYER (8 neurons)<br/>═══════════════<br/>2x Movement direction (dx, dy)<br/>1x Movement speed<br/>1x Eat action<br/>1x Attack action<br/>1x Reproduce action<br/>1x Flee action<br/>1x Social signal"]

        INPUT --> HIDDEN1 --> HIDDEN2 --> OUTPUT
    end
```

## SPH Fluid Dynamics Pipeline

```mermaid
flowchart TD
    subgraph "SPH SIMULATION PIPELINE (Per Tick)"
        A[Build Spatial Hash Grid<br/>Kernel 1: Hash particle positions] --> B[Sort Particles by Hash<br/>Thrust::sort_by_key]
        B --> C[Find Cell Boundaries<br/>Kernel 2: Detect cell start/end]
        C --> D[Compute Density & Pressure<br/>Kernel 3: SPH density summation<br/>Wendland C2 kernel]
        D --> E[Compute Forces<br/>Kernel 4: Pressure + Viscosity +<br/>Surface Tension + Gravity]
        E --> F[Integrate Positions<br/>Kernel 5: Leapfrog integration<br/>with boundary handling]
        F --> G[Boundary Enforcement<br/>Kernel 6: Terrain collision<br/>+ world bounds]
        G --> H{Evaporation/Rain<br/>Coupling?}
        H -->|Yes| I[Climate Exchange Kernel<br/>Water ↔ Atmosphere transfer]
        H -->|No| J[Update Render Buffers]
        I --> J
    end

    subgraph "KERNEL CONFIGURATION"
        K1[Block Size: 256 threads]
        K2[Grid Size: N_particles / 256]
        K3[Shared Memory: Neighbor tile caching<br/>~48KB per block]
        K4[Smoothing Length: h = 0.04]
        K5[Max Neighbors: ~64 per particle]
    end
```


## Genetic Algorithm GPU Pipeline

```mermaid
flowchart TD

TRIGGER{Creature wants to reproduce?} -->|Yes| CHECK[Check Energy greater than Threshold<br/>and Cooldown Complete<br/>and Mate Nearby]

CHECK -->|Pass| SELECT[Selection Kernel<br/>Choose mate from nearby compatible creatures]

SELECT --> CROSSOVER

subgraph CROSSOVER["CROSSOVER KERNEL (GPU)"]
direction TB

P1[Parent 1 Genome - 256 floats]
P2[Parent 2 Genome - 256 floats]

CROSS_TYPE{Crossover Type<br/>from genome gene}

CROSS_TYPE -->|Uniform| UNI[Uniform Crossover<br/>Per-gene random mask]
CROSS_TYPE -->|Single Point| SP[Single Point Crossover<br/>Random split point]
CROSS_TYPE -->|Arithmetic| ARITH[Weighted Average<br/>of parent genes]

UNI --> CHILD[Child Genome]
SP --> CHILD
ARITH --> CHILD

P1 --> CROSS_TYPE
P2 --> CROSS_TYPE

end

CROSSOVER --> MUTATION

subgraph MUTATION["MUTATION KERNEL (GPU)"]
direction TB

M_RATE[Mutation Rate<br/>from parent genome]

M_TYPE1[Gaussian Perturbation<br/>gene plus random gaussian noise]
M_TYPE2[Random Reset<br/>gene becomes random value]
M_TYPE3[Gene Duplication<br/>Copy gene to another slot]
M_TYPE4[Gene Deletion<br/>Set gene to neutral]

CURAND[cuRAND State<br/>per-thread RNG]

CURAND --> M_RATE
M_RATE --> M_TYPE1
M_RATE --> M_TYPE2
M_RATE --> M_TYPE3
M_RATE --> M_TYPE4

end

MUTATION --> SPECIATION

subgraph SPECIATION["SPECIATION KERNEL (GPU)"]
direction TB

DIST[Compute Genetic Distance<br/>to parent species centroid]

THRESH{Distance greater than Speciation Threshold?}

DIST --> THRESH

THRESH -->|Yes| NEW_SPECIES[Assign New Species ID<br/>atomicAdd on species counter]
THRESH -->|No| SAME[Inherit Parent Species ID]

end

SPECIATION --> SPAWN[Spawn Creature Kernel<br/>Initialize position, energy,<br/>neural weights from genome]

SPAWN --> UPDATE_PHYLO[Update Phylogenetic Tree<br/>Record parent to child species link]
```

## Climate System Architecture

```mermaid
flowchart TD
    subgraph "SOLAR RADIATION KERNEL"
        SUN_POS[Sun Position<br/>Day/Night Cycle] --> RADIATION[Compute Solar Flux<br/>Per grid cell based on<br/>latitude + terrain angle]
        RADIATION --> HEAT[Heat Absorption Kernel<br/>Terrain type affects albedo]
    end

    subgraph "ATMOSPHERIC SIMULATION"
        HEAT --> TEMP_DIFF[Temperature Diffusion Kernel<br/>2D Heat Equation<br/>Jacobi Iteration on GPU]
        TEMP_DIFF --> PRESSURE_CALC[Pressure Gradient Kernel<br/>∇P from temperature field]
        PRESSURE_CALC --> WIND_CALC[Wind Vector Update<br/>Navier-Stokes simplified<br/>+ Coriolis effect]
        WIND_CALC --> ADVECTION[Temperature Advection<br/>Semi-Lagrangian method]
    end

    subgraph "WATER CYCLE"
        EVAP[Evaporation Kernel<br/>Water surfaces → humidity] --> HUMID[Humidity Transport<br/>Advected by wind field]
        HUMID --> CONDENSE{Humidity ><br/>Saturation?}
        CONDENSE -->|Yes| CLOUD[Cloud Formation<br/>+ Precipitation Kernel]
        CONDENSE -->|No| CONTINUE[Continue advection]
        CLOUD --> RAIN[Rainfall Distribution<br/>Modify moisture map]
        RAIN --> EROSION[Erosion Kernel<br/>Water flow erodes terrain]
        EROSION --> RIVER[River Formation<br/>Flow accumulation algorithm]
    end

    subgraph "SEASONAL CYCLE"
        SEASON[Season Timer] --> AXIAL[Axial Tilt Factor]
        AXIAL --> SUN_POS
        SEASON --> VEG[Vegetation Growth/Decay<br/>Seasonal modulation]
    end

    ADVECTION --> EVAP
    TEMP_DIFF --> BIOME_UPDATE[Biome Reclassification<br/>Based on temp + moisture]
```

## Rendering Pipeline Architecture

```mermaid
flowchart TD

subgraph INTEROP["CUDA-OpenGL INTEROP PIPELINE"]
    REG[Register OpenGL Buffers<br/>cudaGraphicsGLRegisterBuffer] --> MAP[Map Resources<br/>cudaGraphicsMapResources]
    MAP --> WRITE[CUDA Kernels Write<br/>Directly to GL Buffers]
    WRITE --> UNMAP[Unmap Resources<br/>cudaGraphicsUnmapResources]
    UNMAP --> DRAW[OpenGL Draw Calls]
end

subgraph TERRAIN["TERRAIN RENDERING"]
    T1[Height Map to Vertex Buffer<br/>GPU Kernel] --> T2[Normal Computation Kernel<br/>Sobel filter on heightmap]
    T2 --> T3[Biome Color Assignment<br/>Kernel per vertex]
    T3 --> T4[LOD Selection Kernel<br/>Distance-based tessellation]
end

subgraph CREATURE["CREATURE RENDERING"]
    C1[Frustum Culling Kernel<br/>Test all 2M creatures vs view] --> C2[LOD Assignment Kernel<br/>Near detailed, Far billboard]
    C2 --> C3[Instance Data Packing Kernel<br/>Position, Size, Color, Rotation]
    C3 --> C4[GPU Instanced Rendering<br/>Single draw call, 100K+ instances]
end

subgraph FLUID["FLUID RENDERING"]
    F1[Screen-Space Fluid Rendering<br/>Point sprites to depth] --> F2[Bilateral Filter Kernel<br/>Smooth depth buffer]
    F2 --> F3[Normal from Depth Kernel]
    F3 --> F4[Shading and Refraction]
end

subgraph ATMOS["ATMOSPHERE AND POST-PROCESSING"]
    A1[Cloud Raymarching Kernel]
    A2[Atmospheric Scattering<br/>Rayleigh and Mie]
    A3[Day Night Lighting Transition]
    A4[Bloom and Tone Mapping]
    A5[FXAA Anti-aliasing]

    A1 --> A2 --> A3 --> A4 --> A5
end

T4 --> DRAW
C4 --> DRAW
F4 --> DRAW

DRAW --> A1
```
## Analytics & Phylogenetic Engine

```mermaid
flowchart TD
    subgraph "REAL-TIME ANALYTICS KERNELS"
        POP_COUNT[Population Count per Species<br/>Parallel Histogram Kernel]
        AVG_FITNESS[Average Fitness per Species<br/>Segmented Reduction]
        GENE_DIST[Gene Distribution Analysis<br/>Per-gene histogram across population]
        SPATIAL_DENS[Spatial Density Map<br/>Atomic scatter to grid]
        BIODIVERSITY[Shannon Diversity Index<br/>Parallel log-sum computation]
    end

    subgraph "PHYLOGENETIC TRACKING"
        BIRTH_LOG[Birth Event Logger<br/>Parent species → Child species]
        TREE_BUILD[Tree Node Insertion<br/>Lock-free GPU linked list]
        BRANCH[Branch Length = Time Between<br/>Speciation Events]
        EXTINCT[Extinction Detection<br/>Species population → 0]
    end

    subgraph "EVOLUTIONARY METRICS"
        GEN_SPEED[Generational Turnover Rate]
        MUT_ACCUM[Mutation Accumulation Rate]
        SELECTION_P[Selection Pressure Index<br/>Variance in reproductive success]
        ADAPT_RATE[Adaptation Rate<br/>Fitness change per generation]
    end

    subgraph "VISUALIZATION OUTPUTS"
        PHYLO_VIZ[Phylogenetic Tree Visualization<br/>Force-directed layout on GPU]
        POP_GRAPH[Population Time Series<br/>Ring buffer → line chart]
        HEATMAP_VIZ[Heatmap Overlays<br/>Density, Temperature, etc.]
        GENE_VIZ[Gene Frequency Plots<br/>Per-gene over time]
    end

    POP_COUNT --> POP_GRAPH
    GENE_DIST --> GENE_VIZ
    SPATIAL_DENS --> HEATMAP_VIZ
    BIRTH_LOG --> TREE_BUILD --> PHYLO_VIZ
    AVG_FITNESS --> ADAPT_RATE
    BIODIVERSITY --> POP_GRAPH
```
## Multi-GPU Architecture

```mermaid
graph TB
    subgraph "GPU 0 (Primary)"
        G0_WORLD[World Simulation<br/>Left Half of Map]
        G0_CREATURES[Creatures in Left Region]
        G0_FLUID[Fluid in Left Region]
        G0_RENDER[Full Render Pipeline]
    end

    subgraph "GPU 1 (Secondary)"
        G1_WORLD[World Simulation<br/>Right Half of Map]
        G1_CREATURES[Creatures in Right Region]
        G1_FLUID[Fluid in Right Region]
        G1_ANALYTICS[Full Analytics Pipeline]
    end

    subgraph "BOUNDARY EXCHANGE"
        HALO[Halo Region Exchange<br/>cudaMemcpyPeerAsync]
        MIGRATE[Creature Migration<br/>Cross-GPU transfer when<br/>creature crosses boundary]
        FLUID_BOUND[Fluid Boundary<br/>Ghost particles exchange]
    end

    G0_WORLD <-->|P2P| HALO
    G1_WORLD <-->|P2P| HALO
    G0_CREATURES <-->|NVLink/PCIe| MIGRATE
    G1_CREATURES <-->|NVLink/PCIe| MIGRATE
    G0_FLUID <-->|P2P| FLUID_BOUND
    G1_FLUID <-->|P2P| FLUID_BOUND

    subgraph "LOAD BALANCER"
        LB[Dynamic Region Splitting<br/>Based on creature density]
        LB --> G0_WORLD
        LB --> G1_WORLD
    end
```

## Performance Context
On an NVIDIA RTX 4060 Laptop (Compute Capability 8.9), the engine sustains ~0.35ms per tick (~2,850 Ticks Per Second).
### Benchmark Conditions:
* **Grid**: 256x256 (65,536 cells)
* **Population**: 5,000 interacting agents
* **Neural Topology**: 24 Input → 32 Hidden → 16 Hidden → 8 Output (per agent)
* **Fluid Sim**: Disabled (SPH dominates frametimes when active)
* **Interaction Radius**: 4.0f grid units

## Known Bottlenecks & Hardware Limitations
While highly optimized, the simulation faces realistic hardware constraints under extreme scaling:
* **Memory Bandwidth Limits**: Even with spatial hashing and cell sorting, neighbor querying (combat/mating) results in scattered global memory reads. This makes the interaction phase strictly memory-bound.
* **Warp Divergence**: The neural network engine allows genes to dictate activation functions (ReLU, Tanh, Sigmoid) per organism. If organisms within the same Warp have mutated different activation genes, execution paths diverge, temporarily halving SM efficiency.
* **Atomic Contention**: During high-density population clustering (e.g., around an oasis), atomic additions atomicAdd on vegetation cells and spatial grid bucket indices experience severe thread contention.


## Build Instructions (Windows)
### Prerequisites
* CMake (3.18 or higher)
* Visual Studio 2022 (with "Desktop development with C++" workload)
* NVIDIA CUDA Toolkit (12.x recommended)

### Compilation
Open a PowerShell terminal and run:
```powershell
# Clone the repository
git clone https://github.com/YOUR-USERNAME/GENESIS-CUDA.git
cd GENESIS-CUDA

# Create build directory
mkdir build
cd build

# Configure CMake targeting MSVC and x64 architecture
cmake .. -G "Visual Studio 17 2022" -A x64

# Build the project (Release mode for maximum optimization)
cmake --build . --config Release --parallel
```
## Usage & Scenarios
GENESIS includes a built-in scenario loader that configures the environment, initial population, and evolutionary pressures.

### Run the Default Simulation
```powershell
.\Release\genesis.exe --world-size 256 --creatures 5000
```
### Evolutionary Scenarios

* **The Archipelago** (Speciation focus): High water levels isolate populations.
```powershell
.\Release\genesis.exe --scenario island --world-size 512 --creatures 10000
```

* **The Pandemic** (Immune focus): Dense populations hit by an aggressive SIR-model disease.
```powershell
.\Release\genesis.exe --scenario pandemic --world-size 256 --creatures 10000
```

* **Mass Extinction** (Survival focus): Harsh climate and high energy drain.
```powershell
.\Release\genesis.exe --scenario extinction --world-size 256 --creatures 15000
```
## Running the Test & Benchmark Suites

The project includes automated validation and performance profiling:

```powershell
# Run performance benchmarks (Memory bandwidth, reductions, neural GEMMs)
.\Release\genesis.exe --benchmark

# Run unit tests
.\Release\test_spatial_hash.exe
.\Release\test_genetics.exe
.\Release\test_climate.exe
```