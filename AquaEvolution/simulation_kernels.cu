#include "headers/simulation_kernels.cuh"
#include "headers/helper_math.cuh"
#include <cuda_runtime_api.h>

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>

// ---------------------------------------------------------------- BUCKET SORT FUNCTIONS -----------------------------------------------------------------------

__global__ void calculateAlgaeCellPositions(AquariumSoA aquarium, s_scene scene)
{
	extern __shared__ uint32_t count[];
	count[FISH_COUNT_ID] = *aquarium.fishes.count;
	count[ALGAE_COUNT_ID] = *aquarium.algae.count;
	__syncthreads();

	const uint32_t start_val = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t incr_val = blockDim.x * gridDim.x;

	for (int id = start_val;
		id < scene.cellX*scene.cellY;
		id += incr_val)
	{
		scene.cellBucketSizes[id] = 0;
		for (uint64_t i = 0; i < count[ALGAE_COUNT_ID]; i++)
		{
			uint algaCellId = (uint)(aquarium.algae.positions.x[i] / scene.cellWidth) + (uint)(aquarium.algae.positions.y[i] / scene.cellHieght) * scene.cellX;
			if (id == algaCellId)
			{
				scene.cellArray[calc2DIdx<int>(id, scene.cellBucketSizes[id], scene.pitch)] = i;
				scene.cellBucketSizes[id]++;
				//printf("alga[%u] pos:[%f,%f]\t - cellXid [%u]\t - cellYid [%u] -> cellID: [%u]\n", i, aquarium.algae.positions.x[i], aquarium.algae.positions.y[i], (uint)(aquarium.algae.positions.x[i] / scene.cellWidth), (uint)(aquarium.algae.positions.y[i] / scene.cellHieght), algaCellId);
			}
		}
		//printf("cell [%d] has [%d] algae\n", id, scene.cellBucketSizes[id]);
	}
}
template<typename T>
__device__ int calc2DIdx(int row, int col, size_t pitch)
{
	return row*pitch / sizeof(T) + col;
}

// ---------------------------------------------------------------- FINDING CLOSEST ALGA FUNCTIONS -----------------------------------------------------------------------

__device__ int findClosestAlga(AquariumSoA* aquarium, s_scene* scene, uint32_t fishId, float* distToBeat)
{
	uint2 fishCell = { (uint)(aquarium->fishes.positions.x[fishId] / scene->cellWidth), (uint)(aquarium->fishes.positions.y[fishId] / scene->cellHieght) };

	//check same cell
	int closest_algae_id = findClosestAlgaInCell(aquarium,scene,fishId,fishCell, distToBeat);
	
	// check cells above
	if (fishCell.y > 0)
	{
		// directly above
		closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, {fishCell.x, fishCell.y -1}, distToBeat);
		// left above
		if (fishCell.x > 0)
		{
			closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x - 1, fishCell.y - 1 }, distToBeat);
		}
		// right above
		if (fishCell.x < scene->cellX - 1)
		{
			closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x + 1, fishCell.y - 1 }, distToBeat);
		}
	}
	// check cell below
	if (fishCell.y < scene->cellY - 1)
	{
		// directly below
		closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x, fishCell.y + 1 }, distToBeat);
		// left below
		if (fishCell.x > 0)
		{
			closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x - 1, fishCell.y + 1 }, distToBeat);
		}
		// right below
		if (fishCell.x < scene->cellX - 1)
		{
			closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x + 1, fishCell.y + 1 }, distToBeat);
		}
	}
	// check cell on the left
	if (fishCell.x > 0)
	{
		closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x - 1, fishCell.y }, distToBeat);
	}
	// check cell on the right
	if (fishCell.x < scene->cellX - 1)
	{
		closest_algae_id = findClosestAlgaInCell(aquarium, scene, fishId, { fishCell.x + 1, fishCell.y }, distToBeat);
	}

	return closest_algae_id;
}

__device__ int findClosestAlgaInCell(AquariumSoA* aquarium, s_scene* scene, uint32_t fishId, uint2 cell, float* distToBeat)
{
	int closest_alga_id = -1;
	int cellArrayIdx = cell.x + cell.y * scene->cellX;

	for (int i = 0; i < scene->cellBucketSizes[cellArrayIdx]; i++)
	{
		int alga_id = scene->cellArray[calc2DIdx<int>(cellArrayIdx, i, scene->pitch)];
		float curr_dist = algae_in_sight_dist(aquarium, fishId, alga_id);

		if (curr_dist != -1 && curr_dist < *distToBeat)
		{
			closest_alga_id = alga_id;
			*distToBeat = curr_dist;
		}
	}
	return closest_alga_id;
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

// ---------------------------------------------------------------- MAIN SIMULATION FUNCTIONS -----------------------------------------------------------------------

__global__ void simulate_generation(AquariumSoA aquarium, s_scene scene, curandState* generators)
{
	extern __shared__ uint32_t count[];
	count[FISH_COUNT_ID] = *aquarium.fishes.count;
	count[ALGAE_COUNT_ID] = *aquarium.algae.count;
	__syncthreads();

	const uint32_t start_val = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t incr_val = blockDim.x * gridDim.x;

	for (int j = 0; j < TICKS_PER_GENERATION; ++j)
	{
		fish_decision(&aquarium, &scene, start_val, incr_val);
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
	algae_reproduction(&aquarium, start_val, incr_val, generators);

}

__device__ void fish_decision(AquariumSoA* aquarium, s_scene* scene, uint32_t start_val, uint32_t incr_val)
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
		closest_algae_id = findClosestAlga(aquarium, scene, id, &closest_algae_dist);

		if (closest_algae_id == -1)
		{
			aquarium->fishes.nextDecisions[id] = FishDecisionEnum::NONE;
			return;
		}

		//printf("FISH [%u] FOUND TARGET --- ALGAE[%u]\n", id, closest_algae_id);

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
			energy += ALGAE_ENERGY_VALUE;
			break;
		}
		}

		energy -= aquarium->fishes.stats.energyUsage[id];

		// Check if fish alive
		if (energy <= 0)
		{
			printf("fish[%u] is dead\n", id);
			aquarium->fishes.alives[id] = FishAliveEnum::DEAD;
		}

		aquarium->fishes.currentEnergy[id] = min(energy, aquarium->fishes.stats.maxEnergy[id]);
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

		float energy = aquarium->algae.currentEnergy[id];
		float algae_height = pos_y / AQUARIUM_TOP_BORDER;
		float energyLoss = aquarium->algae.stats.energyUsage[id];
		float energyGain = lerp(0.000075f, 0.00125f, algae_height);
		energy -= energyLoss;
		energy += energyGain;
		if(energy < 0.0f) 
		{
			aquarium->algae.alives[id] = false;
			printf("algae[%u] is dead\n", id);
			continue;
		}

		energy = fminf(energy, 50.0f);

		aquarium->algae.currentEnergy[id] = energy;
	}
}

void fish_reproduction(AquariumSoA* aquarium, uint32_t start_val, uint32_t incr_val, curandState* generators)
{
	uint32_t id;
	extern __shared__ uint32_t count[];
	int fish_count = count[FISH_COUNT_ID];
	curandState generator = generators[start_val];

	// Get data
	int new_index = 0;
	for (int i = 0; i < fish_count; ++i)
	{
		if (aquarium->fishes.alives[i] == FishAliveEnum::ALIVE)
		{
			aquarium->fishes.alives[new_index] = aquarium->fishes.alives[i];
			aquarium->fishes.directionVecs.x[new_index] = aquarium->fishes.directionVecs.x[i];
			aquarium->fishes.directionVecs.y[new_index] = aquarium->fishes.directionVecs.y[i];
			aquarium->fishes.currentEnergy[new_index] = aquarium->fishes.currentEnergy[i];
			aquarium->fishes.interactionEntityIds[new_index] = aquarium->fishes.interactionEntityIds[i];
			aquarium->fishes.nextDecisions[new_index] = aquarium->fishes.nextDecisions[i];
			aquarium->fishes.positions.x[new_index] = aquarium->fishes.positions.x[i];
			aquarium->fishes.positions.y[new_index] = aquarium->fishes.positions.y[i];
			aquarium->fishes.stats.size[new_index] = aquarium->fishes.stats.size[i];

			new_index++;
		}
	}

	int new_fish_count = new_index;
	for (int i = 0; i < new_fish_count; ++i)
	{
		if (new_index + 10 >= Aquarium::maxObjCount)
		{
			*aquarium->fishes.count = new_index;
			return;
		}

		float energy = aquarium->fishes.currentEnergy[i];
		if (energy < 35.0f) continue;
		energy -= 10.0f;
		
		int children_count = (((int)curand(&generator) & INT_MAX) % 10) + 1;
		for (int j = 0; j < children_count; ++j)
		{
			float offx = curand_uniform(&generator) + 1.5f;
			float offy = curand_uniform(&generator) + 1.5f;

			aquarium->fishes.alives[new_index] = FishAliveEnum::ALIVE;
			aquarium->fishes.positions.x[new_index] = aquarium->fishes.positions.x[i] + offx;
			aquarium->fishes.positions.y[new_index] = aquarium->fishes.positions.y[i] + offy;
			aquarium->fishes.directionVecs.x[new_index] = 1.0f;
			aquarium->fishes.directionVecs.y[new_index] = 0.0f;
			aquarium->fishes.stats.size[new_index] = aquarium->fishes.stats.size[i] * 1.1f;
			aquarium->fishes.currentEnergy[new_index] = Fish::ENERGY_INITIAL;
			
			new_index++;
		}
		aquarium->fishes.currentEnergy[i] = energy;
	}

	*aquarium->fishes.count = new_index;
	generators[start_val] = generator;
}

void algae_reproduction(
	AquariumSoA* aquarium,
	uint32_t start_val,
	uint32_t incr_val,
	curandState* generators
)
{
	uint32_t id;
	extern __shared__ uint32_t count[];
	int algae_count = count[ALGAE_COUNT_ID];
	curandState generator = generators[start_val];

	// Get data
	int new_index = 0;
	for (int i = 0; i < algae_count; ++i)
	{
		if (aquarium->algae.alives[i])
		{
			aquarium->algae.positions.x[new_index] = aquarium->algae.positions.x[i];
			aquarium->algae.positions.y[new_index] = aquarium->algae.positions.y[i];
			aquarium->algae.directionVecs.x[new_index] = aquarium->algae.directionVecs.x[i];
			aquarium->algae.directionVecs.y[new_index] = aquarium->algae.directionVecs.y[i];
			aquarium->algae.alives[new_index] = aquarium->algae.alives[i];
			aquarium->algae.currentEnergy[new_index] = aquarium->algae.currentEnergy[i];

			aquarium->algae.stats.size[new_index] = aquarium->algae.stats.size[i];
			aquarium->algae.stats.maxEnergy[new_index] = aquarium->algae.stats.maxEnergy[i];
			aquarium->algae.stats.energyUsage[new_index] = aquarium->algae.stats.energyUsage[i];

			new_index++;
		}
	}

	int new_algae_count = new_index;
	for (int i = 0; i < new_algae_count; ++i)
	{
		if (new_index + 10 >= Aquarium::maxObjCount) break;

		float energy = aquarium->algae.currentEnergy[i];
		if (energy < 28.0f) continue;
		energy -= 5.0f;
		
		int children_count = (((int)curand(&generator) & INT_MAX) % 10) + 1;
		for (int j = 0; j < children_count; ++j)
		{
			float offx = 5.0f * curand_uniform(&generator) - 2.5f;
			float offy = 5.0f * curand_uniform(&generator) - 2.5f;
			float2 vec = { curand_uniform(&generator), curand_uniform(&generator) };
			vec = normalize(vec);
			float2 pos = { 
				clamp(aquarium->algae.positions.x[i] + offx, AQUARIUM_LEFT_BORDER + 0.1f, AQUARIUM_RIGHT_BORDER - 0.1f), 
				clamp(aquarium->algae.positions.y[i] + offy, AQUARIUM_BOTTOM_BORDER + 0.1f, AQUARIUM_TOP_BORDER - 0.1f), 
			};

			aquarium->algae.alives[new_index] = true;
			aquarium->algae.positions.x[new_index] = pos.x;
			aquarium->algae.positions.y[new_index] = pos.y;
			aquarium->algae.directionVecs.x[new_index] = aquarium->algae.directionVecs.x[i];
			aquarium->algae.directionVecs.y[new_index] = aquarium->algae.directionVecs.y[i];
			aquarium->algae.currentEnergy[new_index] = Algae::initialEnergy;

			aquarium->algae.stats.size[new_index] = aquarium->algae.stats.size[i];
			aquarium->algae.stats.maxEnergy[new_index] = aquarium->algae.stats.maxEnergy[i];
			aquarium->algae.stats.energyUsage[new_index] = aquarium->algae.stats.energyUsage[i];
			
			new_index++;
		}
		aquarium->algae.currentEnergy[i] = energy;
	}

	*aquarium->algae.count = new_index;
	generators[start_val] = generator;
}
