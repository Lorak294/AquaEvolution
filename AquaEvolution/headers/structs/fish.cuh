#ifndef FISH_CUH
#define FISH_CUH

#include <vector_types.h>
#include "../structures.cuh"
#include <cstdint>

enum class FishDecisionEnum : int32_t
{ 
	NONE, MOVE, EAT, 
};

enum class FishAliveEnum : int32_t
{
	ALIVE, DEAD
};

struct FishSoA
{
	uint64_t count;

	s_float2 positions;
	s_float2 directionVecs;
	FishAliveEnum* alives;
	float* sizes;
	float* hunger;
	FishDecisionEnum* nextDecisions;
	uint64_t* interactionEntityIds;
	// ... other traits
};

class Fish
{
public:
	// CONST VALUES

	static constexpr float initaialSize				= 1.0f;

	static constexpr float HUNGER_MIN				= 0.0f;
	static constexpr float HUNGER_INITIAL			= 50.0f;
	static constexpr float HUNGER_MAX				= 100.0f;
	static constexpr float HUNGER_CHANGE_PER_TICK	= 0.02f;
	/// If hunger >= HUNGER_MAX => fish dies

public:
	// FIELDS
	float2 position;
	float2 directionVec;
	uint64_t interactionEntityId = -1;
	float size;
	float hunger;
	FishDecisionEnum nextDecision;
	FishAliveEnum is_alive;

	Fish(float2 p, float2 dv, 
		float s, 
		FishAliveEnum alive = FishAliveEnum::ALIVE, 
		FishDecisionEnum decision = FishDecisionEnum::NONE,
		float hunger = HUNGER_INITIAL
	) : 
		position(p), directionVec(dv), 
		size(s), is_alive(alive), 
		nextDecision(decision),
		hunger(hunger)
	{}

	// MEMBER FUNCTIONS
public:

	void writeToDeviceStruct(FishSoA& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.sizes[idx] = size;
		deviceStruct.hunger[idx] = hunger;
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
			deviceStruct.nextDecisions[idx],
			deviceStruct.hunger[idx]
		);
	}
private:
};
#endif // !FISH_CUH