#include "../structures.cuh"
#ifndef ALGAE_CUH
#define ALGAE_CUH


struct AlgaeSoA
{
	uint64_t count;

	s_float2 positions;
	s_float2 directionVecs;
	bool* alives;
	float* sizes;
	// ... other traits
};

class Algae
{
	// FIELDS
public:
	static constexpr float initaialSize = 1.0f;

	float2 position;
	float2 directionVec;
	bool is_alive;
	float size;

	Algae(float2 p, float2 dv, float s, bool alive = true) :
		position(p), directionVec(dv), size(s), is_alive(alive)
	{}

	// MEMBER FUNCTIONS
public:

	void writeToDeviceStruct(AlgaeSoA& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.sizes[idx] = size;
		deviceStruct.directionVecs.x[idx] = directionVec.x;
		deviceStruct.directionVecs.y[idx] = directionVec.y;
		deviceStruct.alives[idx] = is_alive;
	}

	static Algae readFromDeviceStruct(const AlgaeSoA& deviceStruct, int idx)
	{
		return Algae
		(
			{
				deviceStruct.positions.x[idx],
				deviceStruct.positions.y[idx]
			},
			{
				deviceStruct.directionVecs.x[idx],
				deviceStruct.directionVecs.y[idx]
			},
			deviceStruct.sizes[idx],
			deviceStruct.alives[idx]
		);
	}
private:
};

#endif // !ALGAE_CUH