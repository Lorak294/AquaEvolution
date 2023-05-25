#include "headers/simulation_kernels.cuh"
#include <cuda_runtime_api.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>

__global__ void simulate_generation(AquariumSoA aquarium, curandState* generators)
{
	extern __shared__ uint32_t count[]; 
	count[FISH_COUNT_ID] = *aquarium.fishes.count;
	count[ALGAE_COUNT_ID] = *aquarium.algae.count;
	__syncthreads();

	const uint32_t start_val = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t incr_val = blockDim.x * gridDim.x;
	
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

	if (start_val != 0) return;
	fish_reproduction(&aquarium, start_val, incr_val, generators);
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
		float closest_algae_dist_sq = FLT_MAX;

		for (size_t i = 0; i < count[ALGAE_COUNT_ID]; ++i)
		{
			if (!aquarium->algae.alives[i]) continue;

			float algae_x = aquarium->algae.positions.x[i];
			float algae_y = aquarium->algae.positions.y[i];
			float current_dist_sq = (fish_x - algae_x) * (fish_x - algae_x) +
				(fish_y - algae_y) * (fish_y - algae_y);

			if (closest_algae_id == -1 || current_dist_sq < closest_algae_dist_sq)
			{
				closest_algae_dist_sq = current_dist_sq;
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
		bool eat_available = closest_algae_dist_sq < 0.01;
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
		float hunger = aquarium->fishes.hunger[id];
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
			hunger += ALGAE_HUNGER_VALUE;
			break;
		}
		}

		hunger += Fish::HUNGER_CHANGE_PER_TICK;
		//printf("fish[%u].hunger = %f\n", id, hunger);

		// Check if fish alive
		if (hunger >= Fish::HUNGER_MAX)
		{
			printf("fish[%u] is dead\n", id);
			aquarium->fishes.alives[id] = FishAliveEnum::DEAD;
		}

		aquarium->fishes.hunger[id] = hunger;
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

void fish_reproduction(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val, curandState* generators)
{
	uint32_t id;
	extern __shared__ uint32_t count[];
	int fish_count = count[FISH_COUNT_ID];

	// Get data
	int new_index = 0;
	for (int i = 0; i < fish_count; ++i)
	{
		if (aquarium->fishes.alives[i] == FishAliveEnum::ALIVE)
		{
			aquarium->fishes.alives[new_index] = aquarium->fishes.alives[i];
			aquarium->fishes.directionVecs.x[new_index] = aquarium->fishes.directionVecs.x[i];
			aquarium->fishes.directionVecs.y[new_index] = aquarium->fishes.directionVecs.y[i];
			aquarium->fishes.hunger[new_index] = aquarium->fishes.hunger[i];
			aquarium->fishes.interactionEntityIds[new_index] = aquarium->fishes.interactionEntityIds[i];
			aquarium->fishes.nextDecisions[new_index] = aquarium->fishes.nextDecisions[i];
			aquarium->fishes.positions.x[new_index] = aquarium->fishes.positions.x[i];
			aquarium->fishes.positions.y[new_index] = aquarium->fishes.positions.y[i];
			aquarium->fishes.sizes[new_index] = aquarium->fishes.sizes[i];

			new_index++;
		}
	}

	int new_fish_count = new_index;
	for (int i = 0; i < new_fish_count; ++i)
	{
		if (new_index + 5 >= Aquarium::maxObjCount)
		{
			*aquarium->fishes.count = new_index;
			return;
		}

		float hunger = aquarium->fishes.hunger[i];
		if (hunger > Fish::HUNGER_REPRODUCTION_AVAIL) continue;
		hunger += 25.0f;
		
		int children_count = (((int)curand(&generators[start_val]) & 0b01111111111111111111111111111111) % 5) + 1;
		for (int j = 0; j < children_count; ++j)
		{
			float offx = curand_uniform(&generators[start_val]) + 0.5f;
			float offy = curand_uniform(&generators[start_val]) + 0.5f;

			aquarium->fishes.alives[new_index] = FishAliveEnum::ALIVE;
			aquarium->fishes.positions.x[new_index] = aquarium->fishes.positions.x[i] + offx;
			aquarium->fishes.positions.y[new_index] = aquarium->fishes.positions.y[i] + offy;
			aquarium->fishes.directionVecs.x[new_index] = 1.0f;
			aquarium->fishes.directionVecs.y[new_index] = 0.0f;
			aquarium->fishes.sizes[new_index] = aquarium->fishes.sizes[i] * 1.01f;
			aquarium->fishes.hunger[new_index] = Fish::HUNGER_INITIAL;
			
			new_index++;
		}
		aquarium->fishes.hunger[i] = hunger;
	}

	*aquarium->fishes.count = new_index;
}
