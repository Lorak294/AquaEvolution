#ifndef RENDERING_H
#define RENDERING_H

#include "device_launch_parameters.h"

#include "helper_math.cuh"

__global__ void render_GPU(GLubyte* pixels, int maxX, int maxY);
void render_CPU(GLubyte* pixels, int maxX, int maxY);
__host__ __device__ float3 calc_color(float xNorm, float yNorm);
__host__ __device__ float calc_light_power(float xNorm, float yNorm);

// renders scene on bitmap pixels using GPU kernel
__global__ void render_GPU(GLubyte* pixels, int maxX, int maxY)
{
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;

	if (x >= maxX || y >= maxY)
		return;
		
	int pixelIdx = (maxX*y + x) * 3;

	float xNorm = float(x) / maxX;
	float yNorm = float(y) / maxY;

	float3 finalColor = calc_color(xNorm,yNorm);

	pixels[pixelIdx + 0] = (int)(finalColor.x*255.0f);
	pixels[pixelIdx + 1] = (int)(finalColor.y*255.0f);
	pixels[pixelIdx + 2] = (int)(finalColor.z*255.0f);
}

// renders scene on bitmap pixels using CPU single thread
void render_CPU(GLubyte* pixels, int maxX, int maxY)
{
	for (int x = 0; x < maxX; x++)
	{
		for (int y = 0; y < maxY; y++)
		{
			int pixelIdx = (maxX*y + x) * 3;
			
			float xNorm = float(x) / maxX;
			float yNorm = float(y) / maxY;

			float3 finalColor = calc_color(xNorm, yNorm);

			pixels[pixelIdx + 0] = (int)(finalColor.x*255.0f);
			pixels[pixelIdx + 1] = (int)(finalColor.y*255.0f);
			pixels[pixelIdx + 2] = (int)(finalColor.z*255.0f);
		}
	}
}

// returns calculated water/sky color for position
__host__ __device__ float3 calc_color(float xNorm, float yNorm)
{
	const float3 sky = { 0.7f, 0.8f, 1.0f };
	const float3 surfaceWater = { 0.0f, 0.9f, 0.9f };
	const float3 deepWater = { 0.0f, 0.0f, 0.0f };
	const float surfaceOffset = 0.9f;

	if (yNorm > surfaceOffset)
		return { 0.7f, 0.8f, 1.0f };

	float lightFactor = calc_light_power(xNorm, yNorm*surfaceOffset);

	return (surfaceWater - deepWater)*lightFactor;
}

//returns light power strength in given point
__host__ __device__ float calc_light_power(float xNorm, float yNorm)
{
	return yNorm;
}


#endif
