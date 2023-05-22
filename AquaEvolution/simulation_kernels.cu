#include "headers/simulation_kernels.cuh"
#include <cuda_runtime_api.h>

//__global__ void decision_fish(AquariumSoA aquarium, s_scene scene)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.fishes.count) return;
//
//	float fish_x = aquarium.fishes.positions.x[id];
//	float fish_y = aquarium.fishes.positions.y[id];
//
//	// Loop through all algea and find closest one
//	int closest_algae_id = -1;
//	float closest_algae_dist_sq = FLT_MAX;
//
//	for (size_t i = 0; i < aquarium.algae.count; ++i)
//	{
//		if (!aquarium.algae.alives[i]) continue;
//
//		float algae_x = aquarium.algae.positions.x[i];
//		float algae_y = aquarium.algae.positions.y[i];
//		float current_dist_sq = (fish_x - algae_x) * (fish_x - algae_x) +
//			(fish_y - algae_y) * (fish_y - algae_y);
//
//		if (closest_algae_id == -1 || current_dist_sq < closest_algae_dist_sq)
//		{
//			closest_algae_dist_sq = current_dist_sq;
//			closest_algae_id = i;
//		}
//	}
//
//	if (closest_algae_id == -1)
//	{
//		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::NONE;
//		return;
//	}
//
//	// update fishes vector if there is any algea on map
//	float algae_x = aquarium.algae.positions.x[closest_algae_id];
//	float& vec_x = aquarium.fishes.directionVecs.x[id];
//	vec_x = algae_x - fish_x;
//
//	float algae_y = aquarium.algae.positions.y[closest_algae_id];
//	float& vec_y = aquarium.fishes.directionVecs.y[id];
//	vec_y = algae_y - fish_y;
//
//	float d = sqrtf(vec_x * vec_x + vec_y * vec_y);
//	if (d > 0.01f)
//	{
//		vec_x /= d;
//		vec_y /= d;
//	}
//
//	// Check if eating is available
//	bool eat_available = closest_algae_dist_sq < 0.01;
//	if (eat_available)
//	{
//		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::EAT;
//		aquarium.fishes.interactionEntityIds[id] = closest_algae_id;
//		aquarium.algae.alives[closest_algae_id] = false;
//	}
//	else
//	{
//		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::MOVE;
//	}
//}
//
//__global__ void decision_algae(AquariumSoA aquarium, s_scene scene)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.algae.count) return;
//
//	// CALCULATE ALGAE
//
//	// TODO(kutakw): generate random numbers
//	// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types
//	// https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1gf1ba3a7a4a53b2bee1d7c1c7b837c00d
//	// curand_uniform for sobol32
//
//	float vec_x = aquarium.algae.directionVecs.x[id];
//	float vec_y = aquarium.algae.directionVecs.y[id];
//	float denom = sqrtf(vec_x * vec_x + vec_y * vec_y);
//	if (denom >= 0.00001f)
//	{
//		vec_x /= denom;
//		vec_y /= denom;
//	}
//
//	// Bounces
//	// TODO(kutakw): wtf with those boundries
//	float pos_x = aquarium.algae.positions.x[id];
//	float new_pos_x = pos_x + vec_x * 0.01f;
//	if (new_pos_x < -50.0f || new_pos_x >= 150.0f)
//		vec_x *= -1.0f;
//
//	aquarium.algae.directionVecs.x[id] = vec_x;
//
//	float pos_y = aquarium.algae.positions.y[id];
//	float new_pos_y = pos_y + vec_y * 0.01f;
//	if (new_pos_y < -50.0f || new_pos_y >= 150.0f)
//		vec_y *= -1.0f;
//
//	aquarium.algae.directionVecs.y[id] = vec_y;
//}
//
//__global__ void move_fish(AquariumSoA aquarium, s_scene scene)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.fishes.count) return;
//
//	FishDecisionEnum decision = aquarium.fishes.nextDecisions[id];
//	switch (decision)
//	{
//	case FishDecisionEnum::NONE:
//		break;
//	case FishDecisionEnum::MOVE:
//	{
//		// Update FISHES pos 
//		float& pos_x = aquarium.fishes.positions.x[id];
//		float vec_x = aquarium.fishes.directionVecs.x[id];
//		pos_x += vec_x * 0.025f;
//
//		float& pos_y = aquarium.fishes.positions.y[id];
//		float vec_y = aquarium.fishes.directionVecs.y[id];
//		pos_y += vec_y * 0.025f;
//		break;
//	}
//	case FishDecisionEnum::EAT:
//	{
//		// TODO(kutakw): real eating :))
//		printf("%d: EATING algea nr %llu\n", id, aquarium.fishes.interactionEntityIds[id]);
//		return;
//	}
//	}
//}
//
//__global__ void move_algae(AquariumSoA aquarium, s_scene scene)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.algae.count) return;
//
//	if (!aquarium.algae.alives[id]) return;
//
//	float& pos_x = aquarium.algae.positions.x[id];
//	float vec_x = aquarium.algae.directionVecs.x[id];
//	pos_x += vec_x *0.01f;
//
//	float& pos_y = aquarium.algae.positions.y[id];
//	float vec_y = aquarium.algae.directionVecs.y[id];
//	pos_y += vec_y *0.01f;
//}
//
//__global__ void sort_fish(AquariumSoA aquarium, s_scene scene)
//{
//}
//
//__global__ void sort_algae(AquariumSoA aquarium, s_scene scene)
//{
//}
//
//__global__ void new_generation_fish(AquariumSoA aquarium)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.fishes.count) return;
//
//
//}
//
//__global__ void new_generation_algae(AquariumSoA aquarium)
//{
//	// standard parallel approach
//	int id = blockIdx.x * blockDim.x + threadIdx.x;
//	if (id >= aquarium.algae.count) return;
//}

#define FISH_COUNT_ID 0
#define ALGAE_COUNT_ID 1

#define FISH_VELOCITY 0.001f
#define ALGAE_VELOCITY 0.0008f

#define TICKS_PER_GENERATION 1000

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

__device__ void fish_decision(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
	uint32_t id;
	extern __shared__ uint32_t count[]; 
	
	// Fish decision
	for (id = start_val;
		id < count[FISH_COUNT_ID];
		id += incr_val)
	{
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
		if (new_pos_x < -50.0f || new_pos_x >= 150.0f)
			vec_x *= -1.0f;

		aquarium->algae.directionVecs.x[id] = vec_x;

		float pos_y = aquarium->algae.positions.y[id];
		float new_pos_y = pos_y + vec_y * ALGAE_VELOCITY;
		if (new_pos_y < -50.0f || new_pos_y >= 150.0f)
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
		FishDecisionEnum decision = aquarium->fishes.nextDecisions[id];
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
			return;
		}
		}
	}
}

void algae_move(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val)
{
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
