#ifndef SIMULATION_KERNELS_H
#define SIMULATION_KERNELS_H

#include "device_launch_parameters.h"
#include "structs/aquarium.cuh"
#include "scene.cuh"

/// <summary>
/// New approach to kernel execution, no parallel for fishes and algae
/// 1. Fish decision
/// 2. Algae decision
/// 3. Fish move
/// 4. Algae move
/// 5. Fish and Algae sort
/// </summary>
/// <param name="aquarium"></param>
/// <returns></returns>
__global__ void simulate_generation(AquariumSoA aquarium);

__device__ 
void fish_decision(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
void algae_decision(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
void fish_move(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
void algae_move(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

#endif // !SIMULATION_KERNELS_H
