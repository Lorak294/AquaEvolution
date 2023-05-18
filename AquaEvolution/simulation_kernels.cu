#include "headers/simulation_kernels.cuh"
#include "headers/helper_math.cuh"

__global__ void decision_fish(AquariumSoA aquarium, SceneSoA scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.fish.count) return;

	float2 fishPos = aquarium.fish.positions[id];

	// Loop through all algea and find closest one
	int closest_algae_id = -1;
	float closest_algae_dist_sq = FLT_MAX;

	for (size_t i = 0; i < aquarium.algae.count; i++)
	{
		if (!aquarium.algae.alives[i]) continue;


		float2 algaPos = aquarium.algae.positions[i];
		float2 relationVec = fishPos - algaPos;
		float current_dist_sq = dot(relationVec, relationVec);

		if (closest_algae_id == -1 || current_dist_sq < closest_algae_dist_sq)
		{
			closest_algae_dist_sq = current_dist_sq;
			closest_algae_id = i;
		}
	}

	if (closest_algae_id == -1)
	{
		aquarium.fish.nextDecisions[id] = FishDecisionEnum::NONE;
		return;
	}

	// update fishes vector if there is any algea on map
	float2 algaPos = aquarium.algae.positions[closest_algae_id];
	aquarium.fish.directionVecs[id] = normalize(algaPos - fishPos);

	// Check if eating is available
	bool eat_available = closest_algae_dist_sq < 0.01;
	if (eat_available)
	{
		aquarium.fish.nextDecisions[id] = FishDecisionEnum::EAT;
		aquarium.fish.interactionEntityIds[id] = closest_algae_id;
		aquarium.algae.alives[closest_algae_id] = false;
	}
	else
	{
		aquarium.fish.nextDecisions[id] = FishDecisionEnum::MOVE;
	}
}

__global__ void decision_algae(AquariumSoA aquarium, SceneSoA scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.algae.count) return;

	// CALCULATE ALGAE

	// TODO(kutakw): generate random numbers
	// https://docs.nvidia.com/cuda/curand/host-api-overview.html#generator-types
	// https://docs.nvidia.com/cuda/curand/group__DEVICE.html#group__DEVICE_1gf1ba3a7a4a53b2bee1d7c1c7b837c00d
	// curand_uniform for sobol32

	// i'll just leave this here like that - Karol
	float2 vec = normalize(aquarium.algae.directionVecs[id]);
}

__global__ void move_fish(AquariumSoA aquarium, SceneSoA scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.fish.count) return;

	FishDecisionEnum decision = aquarium.fish.nextDecisions[id];
	switch (decision)
	{
	case FishDecisionEnum::NONE:
		break;
	case FishDecisionEnum::MOVE:
	{
		// Update FISHES pos 
		aquarium.fish.positions[id] += aquarium.fish.directionVecs[id] * 0.025f;
		break;
	}
	case FishDecisionEnum::EAT:
	{
		// TODO(kutakw): real eating :))
		printf("%d: EATING algea nr %llu\n", id, aquarium.fish.interactionEntityIds[id]);
		return;
	}
	}
}

__global__ void move_algae(AquariumSoA aquarium, SceneSoA scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.algae.count) return;

	if (!aquarium.algae.alives[id]) return;

	aquarium.algae.positions[id] += aquarium.algae.directionVecs[id] * 0.01f;
}
//
//__global__ void sort_fish(AquariumSoA aquarium, SceneSoA)
//{
//}
//
//__global__ void sort_algae(AquariumSoA aquarium, SceneSoA)
//{
//}