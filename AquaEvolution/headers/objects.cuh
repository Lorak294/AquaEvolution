#ifndef OBJECTS
#define OBJECTS

#include "helper_math.cuh";
#include <thrust/device_vector.h>

// structure of arrays for device (map thrust pointers to normal ones before passing)
struct s_objects {
	float2* positions;
	float2* directionVecs;
	bool* fish; // this field determines object type
	bool* alives;
	float* sizes;
	// ... other traits
};

// class for host
class Object 
{
public:
	static const float initaialSize;

	float2 position;
	float2 directionVec;
	bool alive;
	bool fish;
	float size;

	Object(float2 p, float2 dv, float s, bool fish, bool alive = true) : position(p), directionVec(dv), size(s), fish(fish), alive(alive)
	{

	}

	void writeToDeviceStruct(s_objects& deviceStruct, int idx)
	{
		deviceStruct.positions[idx] = position;
		deviceStruct.alives[idx] = alive;
		deviceStruct.sizes[idx] = size;
		deviceStruct.fish[idx] = fish;
		deviceStruct.directionVecs[idx] = directionVec;
	}

	static Object readFromDeviceStruct(const s_objects& deviceStruct, int idx)
	{
		return Object(
			deviceStruct.positions[idx],
			deviceStruct.directionVecs[idx],
			deviceStruct.sizes[idx],
			deviceStruct.fish[idx],
			deviceStruct.alives[idx]);
	}

	uint hashCell(float cellWidth, float cellHeight)
	{
		return (((uint)(position.x / cellWidth) << 16) | (uint)(position.y / cellHeight));
	}

	static uint cellIdxFromHash(uint hash, int cellsX)
	{
		uint mask = (1 << 16) - 1;
		uint y = hash & mask;
		uint x = hash >> 16;

		return x * cellsX + y;
	}
};

#endif // !OBJECTS
