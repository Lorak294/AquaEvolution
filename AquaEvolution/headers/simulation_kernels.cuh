#ifndef SIMULATION_KERNELS_H
#define SIMULATION_KERNELS_H

#include "device_launch_parameters.h"
#include "structs/aquarium.cuh"
#include "scene.cuh"

/// <summary>
/// Kernel to perform decision phase calculation for fishes. In this kernel it is decided
/// upon wherther fish move in some direction or eat some algae
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void decision_fish(AquariumSoA aquarium, s_scene scene);

/// <summary>
/// Kernel to perform decision phase calculation for algae. 
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void decision_algae(AquariumSoA aquarium, s_scene scene);

/// <summary>
/// Kernel to perfom move phase calculation for fishes.
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void move_fish(AquariumSoA aquarium, s_scene scene);

/// <summary>
/// Kernel to perfom move phase calculation for algae.
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void move_algae(AquariumSoA aquarium, s_scene scene);

/// <summary>
/// Kernel to perform backet/radix/whatever sort on fishes and verify their position in aquarium
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void sort_fish(AquariumSoA aquarium, s_scene);

/// <summary>
/// Kernel to perform backet/radix/whatever sort on algae and verify their position in aquarium
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void sort_algae(AquariumSoA aquarium, s_scene);

__global__ void new_generation_fish(AquariumSoA aquarium);

__global__ void new_generation_algae(AquariumSoA aquarium);

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
//static inline 
void fish_decision(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
//static inline 
void algae_decision(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
//static inline 
void fish_move(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

__device__ 
//static inline 
void algae_move(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val
	);

#endif // !SIMULATION_KERNELS_H
