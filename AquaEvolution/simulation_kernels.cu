#include "headers/simulation_kernels.cuh"
#include <cuda_runtime_api.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>

////////////////////////////// NEUTRAL VALUES ///////////////////////////////

constexpr uint64_t FISH_COUNT_ID			= 0;
constexpr uint64_t ALGAE_COUNT_ID			= 1;
constexpr float ALGAE_ENERGY_VALUE			= 40.f;
constexpr float AQUARIUM_LEFT_BORDER		= 0.f;
constexpr float AQUARIUM_BOTTOM_BORDER		= 0.f;
constexpr float AQUARIUM_RIGHT_BORDER		= 100.f;
constexpr float AQUARIUM_TOP_BORDER			= 100.f;

/////////////////////////////////////////////////////////////////////////////

#ifndef NDEBUG
constexpr float FISH_VELOCITY				= 0.01f;
constexpr float ALGAE_VELOCITY				= 0.008f;
constexpr uint64_t TICKS_PER_GENERATION		= 100;
#else
constexpr float FISH_VELOCITY				= 0.01f;
constexpr float ALGAE_VELOCITY				= 0.008f;
constexpr uint64_t TICKS_PER_GENERATION		= 100;
#endif

__global__ void simulate_generation(AquariumSoA aquarium)
{
	extern __shared__ uint32_t count[]; 
	count[FISH_COUNT_ID] = aquarium.fishes.count;
	count[ALGAE_COUNT_ID] = aquarium.algae.count;
	__syncthreads();

	const uint32_t start_val = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t incr_val = blockDim.x;
	
	for (int j = 0; j < TICKS_PER_GENERATION; ++j)
	{
		fish_decision(&aquarium, start_val, incr_val);
		__syncthreads();

		algae_decision(&aquarium, start_val, incr_val);
		__syncthreads();

		fish_move(&aquarium, start_val, incr_val);
		__syncthreads();

		algae_move(&aquarium, start_val, incr_val);
		__syncthreads();
	}
}

// returns algae distance or -1 if fish cannot see the algae
__device__ float algae_in_sight_dist(AquariumSoA* aquarium, uint32_t fishId, size_t algaId)
{
	float2 algaPos = { aquarium->algae.positions.x[algaId], aquarium->algae.positions.y[algaId] };
	float2 fishPos = { aquarium->fishes.positions.x[fishId], aquarium->fishes.positions.y[fishId] };
	float2 fishVec = { aquarium->fishes.directionVecs.x[fishId], aquarium->fishes.directionVecs.y[fishId] };

	float2 vecToAlga = algaPos - fishPos;

	float dist = length(vecToAlga);

	//check distance
	if (dist > aquarium->fishes.stats.sightDist[fishId])
		return -1.f;

	// check angle
	float cosine = dot(fishVec, vecToAlga/dist);
	if (cosine < aquarium->fishes.stats.sightAngle[fishId])
		return -1.f;

	return dist;
}

__device__ void fish_decision(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
	uint32_t id;
	extern __shared__ uint32_t count[]; 
	
	// Fish decision
	for (id = start_val;
		id < count[FISH_COUNT_ID];
		id += incr_val)
	{
		if (aquarium->fishes.alives[id] == FishAliveEnum::DEAD) continue;

		float fish_x = aquarium->fishes.positions.x[id];
		float fish_y = aquarium->fishes.positions.y[id];

		// Loop through all algea and find closest one
		int closest_algae_id = -1;
		float closest_algae_dist = FLT_MAX;

		for (size_t i = 0; i < count[ALGAE_COUNT_ID]; ++i)
		{
			if (!aquarium->algae.alives[i]) continue;

			float curr_dist = algae_in_sight_dist(aquarium, id, i);
			if (curr_dist == -1.f)
				continue;

			if (closest_algae_id == -1 || curr_dist < closest_algae_dist)
			{
				closest_algae_dist = curr_dist;
				closest_algae_id = i;
			}
		}

		if (closest_algae_id == -1)
		{
			aquarium->fishes.nextDecisions[id] = FishDecisionEnum::NONE;
			return;
		}


		// update fishes vector if there is any algea on map
		float algae_x = aquarium->algae.positions.x[closest_algae_id];
		float& vec_x = aquarium->fishes.directionVecs.x[id];
		vec_x = algae_x - fish_x;

		float algae_y = aquarium->algae.positions.y[closest_algae_id];
		float& vec_y = aquarium->fishes.directionVecs.y[id];
		vec_y = algae_y - fish_y;

		float d = sqrtf(vec_x * vec_x + vec_y * vec_y);
		if (d > 0.01f)
		{
			vec_x /= d;
			vec_y /= d;
		}

		// Check if eating is available
		bool eat_available = closest_algae_dist < 0.01;
		if (eat_available)
		{
			aquarium->fishes.nextDecisions[id] = FishDecisionEnum::EAT;
			aquarium->fishes.interactionEntityIds[id] = closest_algae_id;
			aquarium->algae.alives[closest_algae_id] = false;
		}
		else
		{
			aquarium->fishes.nextDecisions[id] = FishDecisionEnum::MOVE;
		}
	}
}

void algae_decision(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
	uint32_t id;
	extern __shared__ uint32_t count[]; 

	// Algae decision
	for (id = start_val;
		id < count[ALGAE_COUNT_ID];
		id += incr_val)
	{
		if (!aquarium->algae.alives[id]) continue;
		float vec_x = aquarium->algae.directionVecs.x[id];
		float vec_y = aquarium->algae.directionVecs.y[id];
		float denom = sqrtf(vec_x * vec_x + vec_y * vec_y);
		if (denom >= 0.00001f)
		{
			vec_x /= denom;
			vec_y /= denom;
		}

		// Bounces
		// TODO(kutakw): wtf with those boundries
		float pos_x = aquarium->algae.positions.x[id];
		float new_pos_x = pos_x + vec_x * ALGAE_VELOCITY;
		if (new_pos_x < AQUARIUM_LEFT_BORDER || new_pos_x >= AQUARIUM_RIGHT_BORDER)
			vec_x *= -1.0f;

		aquarium->algae.directionVecs.x[id] = vec_x;

		float pos_y = aquarium->algae.positions.y[id];
		float new_pos_y = pos_y + vec_y * ALGAE_VELOCITY;
		if (new_pos_y < AQUARIUM_BOTTOM_BORDER || new_pos_y >= AQUARIUM_TOP_BORDER)
			vec_y *= -1.0f;

		aquarium->algae.directionVecs.y[id] = vec_y;
	}
}

void fish_move(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
	uint32_t id;
	extern __shared__ uint32_t count[]; 

	// Fish move
	for (id = start_val;
		id < count[FISH_COUNT_ID];
		id += incr_val)
	{
		if (aquarium->fishes.alives[id] == FishAliveEnum::DEAD) continue;

		FishDecisionEnum decision = aquarium->fishes.nextDecisions[id];
		float energy = aquarium->fishes.currentEnergy[id];
		switch (decision)
		{
		case FishDecisionEnum::NONE:
			break;
		case FishDecisionEnum::MOVE:
		{
			// Update FISHES pos 
			float& pos_x = aquarium->fishes.positions.x[id];
			float vec_x = aquarium->fishes.directionVecs.x[id];
			pos_x += vec_x * FISH_VELOCITY;

			float& pos_y = aquarium->fishes.positions.y[id];
			float vec_y = aquarium->fishes.directionVecs.y[id];
			pos_y += vec_y * FISH_VELOCITY;
			break;
		}
		case FishDecisionEnum::EAT:
		{
			// TODO(kutakw): real eating :))
			//printf("%d: EATING algea nr %llu\n", id, aquarium->fishes.interactionEntityIds[id]);
			energy += ALGAE_ENERGY_VALUE;
			break;
		}
		}

		energy -= aquarium->fishes.stats.energyUsage[id];
		//printf("fish[%u].hunger = %f\n", id, hunger);

		// Check if fish alive
		if (energy <= 0)
		{
			printf("fish[%u] is dead\n", id);
			aquarium->fishes.alives[id] = FishAliveEnum::DEAD;
		}

		aquarium->fishes.currentEnergy[id] = max(energy, aquarium->fishes.stats.maxEnergy[id]);
	}
}

void algae_move(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
	// TODO(kutakw): generate random numbers
	// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types
	// https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1gf1ba3a7a4a53b2bee1d7c1c7b837c00d
	// curand_uniform for sobol32

	uint32_t id;
	extern __shared__ uint32_t count[];

	// ALGAE MOVE
	for (id = start_val;
		id < count[ALGAE_COUNT_ID];
		id += incr_val)
	{
		if (!aquarium->algae.alives[id]) continue;

		float& pos_x = aquarium->algae.positions.x[id];
		float vec_x = aquarium->algae.directionVecs.x[id];
		pos_x += vec_x * ALGAE_VELOCITY;

		float& pos_y = aquarium->algae.positions.y[id];
		float vec_y = aquarium->algae.directionVecs.y[id];
		pos_y += vec_y * ALGAE_VELOCITY;
	}
}
