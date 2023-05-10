#ifndef SCENE_CUH
#define SCENE_CUH

struct s_scene
{
	int cellX;
	int cellY;
	float cellWidth;
	float cellHieght;

	// array that stores objects sorted by cells  1000x1000 10.000x10.0000
	uint* objCellArray; // x << 16
	// array that stores cellSizes and indexes in the objCellArray
	uint* cellSizeArray;
};

#endif
