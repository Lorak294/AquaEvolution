#ifndef ALGAE
#define ALGAE

#include "structures.cuh";

// structure of arrays for device
struct s_objects {
	s_float2 positions;
	s_float2 directionVecs;
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
	float2 driftingVec;
	bool alive;
	bool fish;
	float size;

	Object(float2 p, float2 dv, float s,bool fish, bool alive = true) : position(p), driftingVec(dv), size(s), fish(fish), alive(alive)
	{

	}

	void writeToDeviceStruct(s_objects& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.alives[idx] = alive;
		deviceStruct.sizes[idx] = size;
		deviceStruct.fish[idx] = fish;
		deviceStruct.directionVecs.x[idx] = driftingVec.x;
		deviceStruct.directionVecs.y[idx] = driftingVec.y;
	}

	static Object readFromDeviceStruct(const s_objects& deviceStruct, int idx)
	{
		return Object(
			{ deviceStruct.positions.x[idx], deviceStruct.positions.y[idx] },
			{ deviceStruct.directionVecs.x[idx], deviceStruct.directionVecs.y[idx] },
			deviceStruct.sizes[idx],
			deviceStruct.fish[idx],
			deviceStruct.alives[idx]);
	}
};

#endif // !ALGAE
