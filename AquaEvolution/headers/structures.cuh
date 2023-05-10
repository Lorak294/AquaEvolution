#ifndef STRUCTURES_CUH
#define STRUCTURES_CUH

#include "helper_math.cuh"

struct s_float2 
{
	float* x;
	float* y;
};

struct s_scene
{
	int cellX;
	int cellY;
	float cellWidth;
	float cellHieght;

	// array that stores objects sorted by cells 
	uint* objCellArray;
	// array that stores cellSizes and indexes in the objCellArray
	uint* cellSizeArray;
};

#endif // !STRUCTURES_CUH

