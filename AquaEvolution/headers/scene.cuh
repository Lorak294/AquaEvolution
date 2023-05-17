#ifndef SCENE_CUH
#define SCENE_CUH

// this struct is only for passing data to GPU kernel (kernel requires array pointers istead of thrust vectors)
struct s_scene
{
	int cellX;
	int cellY;
	float cellWidth;
	float cellHieght;
	int objCount;

	//pointer to array that stores objects sorted by cells  1000x1000 10.000x10.0000
	uint* objCellArray; // x << 16
	//pointer to array that stores cellSizes and indexes in the objCellArray
	uint* cellSizeArray;
};

#endif
