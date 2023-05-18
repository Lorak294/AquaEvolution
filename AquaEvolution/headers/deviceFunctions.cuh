#ifndef DEVICEFUNCTIONS_CUH
#define DEVICEFUNCTIONS_CUH

#include "device_launch_parameters.h"
#include "helper_math.cuh"

__host__ float3 calc_color(float xNorm, float yNorm);
__host__ float calc_light_power(float xNorm, float yNorm);
__device__ float3 device_calc_color(float xNorm, float yNorm);
__device__ float device_calc_light_power(float xNorm, float yNorm);

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
