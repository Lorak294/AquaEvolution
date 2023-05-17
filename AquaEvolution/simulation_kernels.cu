#include "headers/simulation_kernels.cuh"

__global__ void decision_fish(AquariumSoA aquarium, s_scene scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.fishes.count) return;

	float fish_x = aquarium.fishes.positions.x[id];
	float fish_y = aquarium.fishes.positions.y[id];

	// Loop through all algea and find closest one
	int closest_algae_id = -1;
	float closest_algae_dist_sq = FLT_MAX;

	for (size_t i = 0; i < aquarium.algae.count; ++i)
	{
		if (!aquarium.algae.alives[i]) continue;

		float algae_x = aquarium.algae.positions.x[i];
		float algae_y = aquarium.algae.positions.y[i];
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
		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::NONE;
		return;
	}

	// update fishes vector if there is any algea on map
	float algae_x = aquarium.algae.positions.x[closest_algae_id];
	float& vec_x = aquarium.fishes.directionVecs.x[id];
	vec_x = algae_x - fish_x;

	float algae_y = aquarium.algae.positions.y[closest_algae_id];
	float& vec_y = aquarium.fishes.directionVecs.y[id];
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
		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::EAT;
		aquarium.fishes.interactionEntityIds[id] = closest_algae_id;
		aquarium.algae.alives[closest_algae_id] = false;
	}
	else
	{
		aquarium.fishes.nextDecisions[id] = FishDecisionEnum::MOVE;
	}
}

__global__ void decision_algae(AquariumSoA aquarium, s_scene scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.algae.count) return;

	// CALCULATE ALGAE

	// TODO(kutakw): generate random numbers
	// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types
	// https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1gf1ba3a7a4a53b2bee1d7c1c7b837c00d
	// curand_uniform for sobol32

	float vec_x = aquarium.algae.directionVecs.x[id];
	float vec_y = aquarium.algae.directionVecs.y[id];
	float denom = sqrtf(vec_x * vec_x + vec_y * vec_y);
	if (denom >= 0.00001f)
	{
		vec_x /= denom;
		vec_y /= denom;
	}
}

__global__ void move_fish(AquariumSoA aquarium, s_scene scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.fishes.count) return;

	FishDecisionEnum decision = aquarium.fishes.nextDecisions[id];
	switch (decision)
	{
	case FishDecisionEnum::NONE:
		break;
	case FishDecisionEnum::MOVE:
	{
		// Update FISHES pos 
		float& pos_x = aquarium.fishes.positions.x[id];
		float vec_x = aquarium.fishes.directionVecs.x[id];
		pos_x += vec_x * 0.025f;

		float& pos_y = aquarium.fishes.positions.y[id];
		float vec_y = aquarium.fishes.directionVecs.y[id];
		pos_y += vec_y * 0.025f;
		break;
	}
	case FishDecisionEnum::EAT:
	{
		// TODO(kutakw): real eating :))
		printf("%d: EATING algea nr %llu\n", id, aquarium.fishes.interactionEntityIds[id]);
		return;
	}
	}
}

__global__ void move_algae(AquariumSoA aquarium, s_scene scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.algae.count) return;

	if (!aquarium.algae.alives[id]) return;

	float& pos_x = aquarium.algae.positions.x[id];
	float vec_x = aquarium.algae.directionVecs.x[id];
	pos_x += vec_x * 0.01f;

	float& pos_y = aquarium.algae.positions.y[id];
	float vec_y = aquarium.algae.directionVecs.y[id];
	pos_y += vec_y * 0.01f;
}

__global__ void sort_fish(AquariumSoA aquarium, s_scene)
{
}

__global__ void sort_algae(AquariumSoA aquarium, s_scene)
{
}