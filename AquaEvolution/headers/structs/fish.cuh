#ifndef FISH_CUH
#define FISH_CUH

#include <vector_types.h>
#include "../structures.cuh"
#include <cstdint>

enum class FishDecisionEnum 
{ 
	NONE, MOVE, EAT, 
};

struct FishSoA
{
	uint64_t count;

	s_float2 positions;
	s_float2 directionVecs;
	bool* alives;
	float* sizes;
	FishDecisionEnum* nextDecisions;
	uint64_t* interactionEntityIds;
	// ... other traits
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

	void writeToDeviceStruct(FishSoA& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.sizes[idx] = size;
		deviceStruct.directionVecs.x[idx] = directionVec.x;
		deviceStruct.directionVecs.y[idx] = directionVec.y;
		deviceStruct.nextDecisions[idx] = nextDecision;
		deviceStruct.alives[idx] = is_alive;
	}

	static Fish readFromDeviceStruct(const FishSoA& deviceStruct, int idx)
	{
		return Fish
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
			deviceStruct.alives[idx],
			deviceStruct.nextDecisions[idx]
		);
	}
private:
};
#endif // !FISH_CUH