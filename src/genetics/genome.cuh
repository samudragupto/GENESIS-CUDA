#ifndef GENOME_CUH
#define GENOME_CUH

#include <cuda_runtime.h>
#include "../core/constants.cuh"

extern __constant__ int d_gene_permutation[512];

void initGenomeConstants();

#endif