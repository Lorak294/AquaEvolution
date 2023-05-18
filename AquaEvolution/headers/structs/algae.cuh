#ifndef ALGAE_CUH
#define ALGAE_CUH
#include "../structures.cuh"
#include <thrust/device_vector.h>


struct AlgaeSoA
{
	uint64_t count;
	float2* positions;
	float2* directionVecs;
	bool* alives;
	float* sizes;
	// ... other traits
};

struct AlgaeThrustContainer
{
	thrust::device_vector<float2> positions;
	thrust::device_vector<float2> directionVecs;
	thrust::device_vector<bool> alives;
	thrust::device_vector<float> sizes;

	//possibly others
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

	void writeToDevice(AlgaeThrustContainer& thrustAlgae, int idx)
	{
		thrustAlgae.positions[idx] = position;
		thrustAlgae.sizes[idx] = size;
		thrustAlgae.directionVecs[idx] = directionVec;
		thrustAlgae.alives[idx] = is_alive;
	}

	static Algae readFromDevice(const AlgaeThrustContainer& thrustAlgae, int idx)
	{
		return Algae
		(
			thrustAlgae.positions[idx],
			thrustAlgae.directionVecs[idx],
			thrustAlgae.sizes[idx],
			thrustAlgae.alives[idx]
		);
	}

	uint calcCellHash(float cellWidth, float cellHeight)
	{
		return ((uint)(position.x / cellWidth) + ((uint)(position.y / cellHeight) << 16));
	}

	uint calcCellIdx(float cellWidth, float cellHeight,uint cellX)
	{
		return (uint)(position.x / cellWidth) + ((uint)(position.y / cellHeight) * cellX);
	}
private:
};

#endif // !ALGAE_CUH