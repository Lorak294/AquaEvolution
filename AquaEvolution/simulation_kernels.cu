#include "headers/simulation_kernels.cuh"
#include "headers/helper_math.cuh"

#define AQUARIUMWIDTH 100
#define AQUARIUHEIGHT 100

__device__ uint calcCellIdx(float2 pos, const SceneSoA& scene)
{
	return (uint)(pos.x/ scene.cellWidth) + ((uint)(pos.y / scene.cellHeight) * scene.cellX);
}

__device__ int findClosestAlgaInCell(uint cellStartIdx, uint cellSize, const AquariumSoA& aquarium, float2 fishPos, float& closest_algae_dist_sq)
{
	int closest_algae_id = -1;

	for (size_t i = 0; i < cellSize; i++)
	{
		if (!aquarium.algae.alives[cellStartIdx + i]) continue;

		float2 algaPos = aquarium.algae.positions[cellStartIdx + i];
		float2 relationVec = fishPos - algaPos;
		float current_dist_sq = dot(relationVec, relationVec);

		if (closest_algae_id == -1 || current_dist_sq < closest_algae_dist_sq)
		{
			closest_algae_dist_sq = current_dist_sq;
			closest_algae_id = i;
		}
	}
	return closest_algae_id;
}

__device__ int findClosestAlga(const AquariumSoA& aquarium,const SceneSoA& scene, float2 fishPos, float& closest_algae_dist_sq)
{
	int closest_algae_id = -1;
	int myCellIdx = calcCellIdx(fishPos, scene);
	int myCellXCoord = myCellIdx % scene.cellX;

	//printf("---- FOR FISH POS(%f,%f) CELLIDX(%d) CELLXCOORD(%d)\n", fishPos.x,fishPos.y,myCellIdx,myCellXCoord);

	// check same cell
	//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx);
	closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx], scene.cellIndexesAlgae[myCellIdx], aquarium, fishPos, closest_algae_dist_sq);
	// check cells above
	if (myCellIdx - scene.cellX > 0)
	{
		// directly above
		//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx - scene.cellX);
		closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx - scene.cellX], scene.cellIndexesAlgae[myCellIdx - scene.cellX], aquarium, fishPos, closest_algae_dist_sq);
		// left above
		if (myCellXCoord > 0)
		{
			//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx - scene.cellX - 1);
			closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx - scene.cellX - 1], scene.cellIndexesAlgae[myCellIdx - scene.cellX - 1], aquarium, fishPos, closest_algae_dist_sq);
		}
		// right above
		if (myCellXCoord < scene.cellX - 1)
		{
			//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx - scene.cellX + 1);
			closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx - scene.cellX + 1], scene.cellIndexesAlgae[myCellIdx - scene.cellX + 1], aquarium, fishPos, closest_algae_dist_sq);
		}
	}
	// check cell below
	if (myCellIdx + scene.cellX < scene.cellX*scene.cellY)
	{
		// directly below
		//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx + scene.cellX);
		closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx + scene.cellX], scene.cellIndexesAlgae[myCellIdx + scene.cellX], aquarium, fishPos, closest_algae_dist_sq);
		// left below
		if (myCellXCoord > 0)
		{
			//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx + scene.cellX - 1);
			closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx + scene.cellX - 1], scene.cellIndexesAlgae[myCellIdx + scene.cellX - 1], aquarium, fishPos, closest_algae_dist_sq);
		}
		// right below
		if (myCellXCoord < scene.cellX - 1)
		{
			//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx + scene.cellX + 1);
			closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx + scene.cellX + 1], scene.cellIndexesAlgae[myCellIdx + scene.cellX + 1], aquarium, fishPos, closest_algae_dist_sq);
		}
	}
	// check cell on the left
	if (myCellXCoord > 0)
	{
		//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx - 1);
		closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx - 1], scene.cellIndexesAlgae[myCellIdx - 1], aquarium, fishPos, closest_algae_dist_sq);
	}
	// check cell on the right
	if (myCellXCoord < scene.cellX -1)
	{
		//printf("[%d] searching inc cell %d\n", myCellIdx, myCellIdx + 1);
		closest_algae_id = findClosestAlgaInCell(scene.cellSizesAlgae[myCellIdx + 1], scene.cellIndexesAlgae[myCellIdx + 1], aquarium, fishPos, closest_algae_dist_sq);
	}

	return closest_algae_id;
}


__global__ void decision_fish(AquariumSoA aquarium, SceneSoA scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.fish.count) return;

	float2 fishPos = aquarium.fish.positions[id];

	// Loop through all algea and find closest one
	int closest_algae_id = -1;
	float closest_algae_dist_sq = FLT_MAX;

	closest_algae_id = findClosestAlga(aquarium, scene, fishPos, closest_algae_dist_sq);

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
		float2 newPos = aquarium.fish.positions[id] + aquarium.fish.directionVecs[id]* 0.05f;
		if (newPos.x > AQUARIUMWIDTH || newPos.x < 0)
			aquarium.fish.directionVecs[id].x *= -1;
		if (newPos.y > AQUARIUHEIGHT || newPos.y < 0)
			aquarium.fish.directionVecs[id].y *= -1;
		aquarium.fish.positions[id] += aquarium.fish.directionVecs[id] * 0.05f;
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

	float2 newPos = aquarium.algae.positions[id] + aquarium.algae.directionVecs[id] * 0.05f;
	if (newPos.x > AQUARIUMWIDTH || newPos.x < 0)
		aquarium.algae.directionVecs[id].x *= -1;
	if (newPos.y > AQUARIUHEIGHT || newPos.y < 0)
		aquarium.algae.directionVecs[id].y *= -1;

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