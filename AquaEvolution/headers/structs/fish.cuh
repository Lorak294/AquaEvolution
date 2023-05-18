#ifndef FISH_CUH
#define FISH_CUH

#include <vector_types.h>
#include "../structures.cuh"
#include <cstdint>
#include <thrust/device_vector.h>

enum class FishDecisionEnum
{
	NONE, MOVE, EAT,
};

struct FishSoA
{
	uint64_t count;
	float2* positions;
	float2* directionVecs;
	bool* alives;
	float* sizes;
	FishDecisionEnum* nextDecisions;
	uint64_t* interactionEntityIds;
	// ... other traits
};

struct FishThrustContainer
{
	thrust::device_vector<float2> positions;
	thrust::device_vector<float2> directionVecs;
	thrust::device_vector<bool> alives;
	thrust::device_vector<float> sizes;
	thrust::device_vector<FishDecisionEnum> nextDecisions;
	thrust::device_vector<uint64_t> interactionEntityIds;
	// possibly other traits
};

class Fish
{
	// FIELDS
public:
	static constexpr float initaialSize = 1.0f;

	float2 position;
	float2 directionVec;
	bool is_alive;
	float size;
	FishDecisionEnum nextDecision;
	uint64_t interactionEntityId = -1;

	Fish(float2 p, float2 dv, float s, bool alive = true, FishDecisionEnum decision = FishDecisionEnum::NONE) :
		position(p), directionVec(dv), size(s), is_alive(alive), nextDecision(decision)
	{}

	// MEMBER FUNCTIONS
public:

	void writeToDevice(FishThrustContainer& thrustFish, int idx)
	{
		thrustFish.positions[idx] = position;
		thrustFish.directionVecs[idx] = directionVec;
		thrustFish.alives[idx] = is_alive;
		thrustFish.interactionEntityIds[idx] = interactionEntityId;
		thrustFish.sizes[idx] = size;
		thrustFish.nextDecisions[idx] = nextDecision;
	}

	static Fish readFromDevice(const FishThrustContainer& thrustFish, int idx)
	{
		return Fish
		(
			thrustFish.positions[idx],
			thrustFish.directionVecs[idx],
			thrustFish.sizes[idx],
			thrustFish.alives[idx],
			thrustFish.nextDecisions[idx]
		);
	}

	uint calcCellHash(float cellWidth, float cellHeight)
	{
		return ((uint)(position.x / cellWidth) + ((uint)(position.y / cellHeight) << 16));
	}

	uint calcCellIdx(float cellWidth, float cellHeight, uint cellX)
	{
		return (uint)(position.x / cellWidth) + ((uint)(position.y / cellHeight) * cellX);
	}
private:
};
#endif // !FISH_CUH