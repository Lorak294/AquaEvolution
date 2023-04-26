#ifndef ALGAE
#define ALGAE

#include "structures.cuh";

// structure of arrays for device
struct s_algae {
	s_float2 positions;
	bool* alives;
	float* sizes;
	s_float2 driftingVecs;
};

// class for host
class alga 
{
public:
	static const float initaialSize;

	float2 position;
	float2 driftingVec;
	bool alive;
	float size;

	alga(float2 p, float2 dv, float s, bool alive = true) : position(p), driftingVec(dv), size(s), alive(alive)
	{

	}

	void writeToDeviceStruct(s_algae& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.alives[idx] = alive;
		deviceStruct.sizes[idx] = size;
		deviceStruct.driftingVecs.x[idx] = driftingVec.x;
		deviceStruct.driftingVecs.y[idx] = driftingVec.y;
	}

	static alga readFromDeviceStruct(const s_algae& deviceStruct, int idx)
	{
		return alga(
			{ deviceStruct.positions.x[idx], deviceStruct.positions.y[idx] },
			{ deviceStruct.driftingVecs.x[idx], deviceStruct.driftingVecs.y[idx] },
			deviceStruct.sizes[idx], 
			deviceStruct.alives[idx]);
	}
};

#endif // !ALGAE
