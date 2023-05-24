#ifndef SIMULATION_KERNELS_H
#define SIMULATION_KERNELS_H

#include "device_launch_parameters.h"
#include "structs/aquarium.cuh"
#include "scene.cuh"
#include <curand_kernel.h>

////////////////////////////// NEUTRAL VALUES ///////////////////////////////

constexpr uint64_t FISH_COUNT_ID			= 0;
constexpr uint64_t ALGAE_COUNT_ID			= 1;
constexpr float ALGAE_HUNGER_VALUE			= -40.f;
constexpr float AQUARIUM_LEFT_BORDER		= 0.f;
constexpr float AQUARIUM_BOTTOM_BORDER		= 0.f;
constexpr float AQUARIUM_RIGHT_BORDER		= 100.f;
constexpr float AQUARIUM_TOP_BORDER			= 100.f;
constexpr uint64_t GENERATORS_COUNT = 2048;

/////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
constexpr float FISH_VELOCITY				= 0.01f;
constexpr float ALGAE_VELOCITY				= 0.008f;
constexpr uint64_t TICKS_PER_GENERATION		= 100;
#else
constexpr float FISH_VELOCITY				= 0.005f;
constexpr float ALGAE_VELOCITY				= 0.0001f;
constexpr uint64_t TICKS_PER_GENERATION		= 100;
#endif

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
__global__ void simulate_generation(AquariumSoA aquarium, curandState* generators);

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

__device__
void fish_reproduction(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val,
	curandState* generators
);

#endif // !SIMULATION_KERNELS_H
