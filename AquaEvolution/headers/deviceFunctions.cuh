#ifndef DEVICEFUNCTIONS_CUH
#define DEVICEFUNCTIONS_CUH

#include "device_launch_parameters.h"
#include "aquarium.cuh"
#include "scene.cuh"
#include "helper_math.cuh"

__global__ void simulateGeneration(s_aquarium aquarium, s_scene scene);
__global__ void sortByCells(s_aquarium aquarium, s_scene scene);
__global__ void performMovement(s_aquarium aquarium, s_scene scene);
__global__ void setVectorsAndEat(s_aquarium aquarium, s_scene scene);

__device__ void calc_fish_vector(s_aquarium* aquarium, s_scene* scene, int id);
__device__ void calc_algae_vector(s_aquarium* aquarium, s_scene* scene, int id);

__host__ float3 calc_color(float xNorm, float yNorm);
__host__ float calc_light_power(float xNorm, float yNorm);
__device__ float3 device_calc_color(float xNorm, float yNorm);
__device__ float device_calc_light_power(float xNorm, float yNorm);

// main kernel function
__global__ void simulateGeneration(s_aquarium aquarium, s_scene scene)
{
	// standard parallel approach
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= aquarium.objectsCount) return;

	// fish vectors
	bool is_fish = aquarium.objects.fish[id];
	if (is_fish) {
		float& fish_x = aquarium.objects.positions.x[id];
		float& fish_y = aquarium.objects.positions.y[id];

		// Loop through all algea and find closest one
		int closest_algae_id = -1;
		float closest_algae_dist_sq = FLT_MAX;

		for (size_t i = 0; i < aquarium.objectsCount; ++i) {
			if (aquarium.objects.fish[i]) continue;

			float algae_x = aquarium.objects.positions.x[i];
			float algae_y = aquarium.objects.positions.y[i];
			float current_dist_sq = (fish_x - algae_x) * (fish_x - algae_x) +
				(fish_y - algae_y) * (fish_y - algae_y);

			if (closest_algae_id != -1 && current_dist_sq >= closest_algae_dist_sq) continue;
			closest_algae_dist_sq = current_dist_sq;
			closest_algae_id = i;
		}
		bool eat_available = closest_algae_dist_sq < 0.01;

		// update fishes vector if there is any algea on map
		if (closest_algae_id != -1) {
			float algae_x = aquarium.objects.positions.x[closest_algae_id];
			float& vec_x = aquarium.objects.directionVecs.x[id];
			vec_x = algae_x - fish_x;

			float algae_y = aquarium.objects.positions.y[closest_algae_id];
			float& vec_y = aquarium.objects.directionVecs.y[id];
			vec_y = algae_y - fish_y;

			float d = sqrtf(vec_x * vec_x + vec_y * vec_y);
			if (d > 0.01) {
				vec_x /= d;
				vec_y /= d;
				//printf("d: %f, dist: %f\n", d, vec_x * vec_x + vec_y * vec_y);
			}
		}

		if (eat_available) {
			printf("%d: EATING algea nr %d\n", id, closest_algae_id);
			return;
		}

		// Update FISHES pos 
		float& pos_x = aquarium.objects.positions.x[id];
		float& vec_x = aquarium.objects.directionVecs.x[id];
		pos_x += vec_x * 0.025f;

		float& pos_y = aquarium.objects.positions.y[id];
		float& vec_y = aquarium.objects.directionVecs.y[id];
		pos_y += vec_y * 0.025f;
		//printf("%d: pos: (%f, %f)\n", id, pos_x, pos_y);
		//printf("%d: vec: (%f, %f)\n", id, vec_x, vec_y);
	} else {
		// CALCULATE ALGAE

		float& pos_x = aquarium.objects.positions.x[id];
		pos_x += 0.01f;
		float& pos_y = aquarium.objects.positions.y[id];
		pos_y += 0.01f;
	}



	//float* pos_x = &aquarium.objects.positions.x[i];
	//float* pos_y = &aquarium.objects.positions.y[i];

	//float* vec_x = &aquarium.objects.directionVecs.x[i];
	//float* vec_y = &aquarium.objects.directionVecs.y[i];

	//if(i == 0)
	//	printf("%d: (%f, %f)\n", i, *pos_x, *pos_y);

	//// just fckin add those goddamn values
	//*pos_x += 0.1f;
	//*pos_y += 0.1f;
}

__global__ void sortByCells(s_aquarium aquarium, s_scene scene)
{
}
__global__ void performMovement(s_aquarium aquarium, s_scene scene)
{

}
__global__ void setVectorsAndEat(s_aquarium aquarium, s_scene scene)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= aquarium.objectsCount) return;

	bool is_fish = aquarium.objects.fish[i];
	if (is_fish)
		calc_fish_vector(&aquarium, &scene, i);
	else
		calc_algae_vector(&aquarium, &scene, i);
}

__device__ void calc_fish_vector(s_aquarium* aquarium, s_scene* scene, int id)
{
}

__device__ void calc_algae_vector(s_aquarium* aquarium, s_scene* scene, int id)
{
}

// returns calculated water/sky color for position
__host__  float3 calc_color(float xNorm, float yNorm)
{
	const float3 sky = { 0.7f, 0.8f, 1.0f };
	const float3 surfaceWater = { 0.0f, 0.9f, 0.9f };
	const float3 deepWater = { 0.0f, 0.0f, 0.0f };
	const float surfaceOffset = 0.9f;
 
	if (yNorm > surfaceOffset)
		return sky;

	float lightFactor = calc_light_power(xNorm, yNorm*surfaceOffset);

	return (surfaceWater - deepWater)*lightFactor;
}

// returns calculated water/sky color for position
__device__ float3 device_calc_color(float xNorm, float yNorm)
{
	const float3 sky = { 0.7f, 0.8f, 1.0f };
	const float3 surfaceWater = { 0.0f, 0.9f, 0.9f };
	const float3 deepWater = { 0.0f, 0.0f, 0.0f };
	const float surfaceOffset = 0.9f;

	if (yNorm > surfaceOffset)
		return sky;

	float lightFactor = device_calc_light_power(xNorm, yNorm*surfaceOffset);

	return (surfaceWater - deepWater)*lightFactor;
}

//returns light power strength in given point
__host__ float calc_light_power(float xNorm, float yNorm)
{
	return yNorm;
}

__device__ float device_calc_light_power(float xNorm, float yNorm)
{
	return yNorm;
}


#endif
