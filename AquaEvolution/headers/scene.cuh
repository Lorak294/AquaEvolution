#ifndef SCENE_CUH
#define SCENE_CUH

struct s_scene
{
	int cellX;
	int cellY;
	float cellWidth;
	float cellHieght;

	// algae positions in cells 
	int* cellArray;
	size_t pitch;
	int* cellBucketSizes;
};

#endif
