#ifndef SCENE_CUH
#define SCENE_CUH

// this struct is only for passing data to GPU kernel (kernel requires array pointers istead of thrust vectors)
struct SceneThrustContainer
{
	// arrays that store objects sorted by cells
	thrust::device_vector<uint> fishCellPositioning;
	thrust::device_vector<uint> algaeCellPositioning;
	// arrays that store cellSizes and indexes of the above
	thrust::device_vector<uint> cellSizesFish;
	thrust::device_vector<uint> cellSizesAlgae;
	thrust::device_vector<uint> cellIndexesFish;
	thrust::device_vector<uint> cellIndexesAlgae;
};

struct SceneSoA
{
	// pointers to arrays that store objects sorted by cells  => may even be unnecesarry to pass to kernel
	uint* fishCellPositioning;
	uint* algaeCellPositioning;
	// pointers to array that store cellSizes and indexes of the above
	uint* cellSizesFish;
	uint* cellSizesAlgae;
	uint* cellIndexesFish;
	uint* cellIndexesAlgae;
};

#endif
