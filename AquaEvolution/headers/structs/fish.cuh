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

struct StatsSoA
{
	float* size;
	float* sightDist;
	float* sightAngle; // -1 to 1
	float* maxEnergy;
	float* energyUsage;
};

struct FishSoA
{
	uint64_t count;

	s_float2 positions;
	s_float2 directionVecs;
	FishAliveEnum* alives;
	float* currentEnergy;
	FishDecisionEnum* nextDecisions;
	uint64_t* interactionEntityIds;

	StatsSoA stats;
};

struct Stats
{
	float size;
	float sightDist;
	float sightAngle; // -1 to 1
	float maxEnergy;
	float energyUsage;

	// initial values constructor
	Stats() : size(0.5f), sightDist(10.f), sightAngle(0.f), maxEnergy(50.f), energyUsage(0.01f)
	{

	}
};

class Fish
{
public:
	// CONST VALUES

	static constexpr float initaialSize				= 0.5f;

	//static constexpr float HUNGER_MIN				= 0.0f;
	static constexpr float INITIAL_ENERGY			= 25.0f;
	//static constexpr float HUNGER_MAX				= 100.0f;
	//static constexpr float HUNGER_CHANGE_PER_TICK	= 0.02f;
	/// If hunger >= HUNGER_MAX => fish dies

public:
	// FIELDS
	float2 position;
	float2 directionVec;
	uint64_t interactionEntityId = -1;
	FishDecisionEnum nextDecision;
	FishAliveEnum is_alive;
	float currentEnergy;

	Stats stats;

	Fish(float2 p, float2 dv,
		Stats st,
		FishAliveEnum alive = FishAliveEnum::ALIVE, 
		FishDecisionEnum decision = FishDecisionEnum::NONE,
		float energy = INITIAL_ENERGY
	) : 
		position(p), directionVec(dv), 
		stats(st), is_alive(alive), 
		nextDecision(decision),
		currentEnergy(energy)
	{}

	// MEMBER FUNCTIONS
public:

	void writeToDeviceStruct(FishSoA& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.currentEnergy[idx] = currentEnergy;
		deviceStruct.directionVecs.x[idx] = directionVec.x;
		deviceStruct.directionVecs.y[idx] = directionVec.y;
		deviceStruct.nextDecisions[idx] = nextDecision;
		deviceStruct.alives[idx] = is_alive;

		deviceStruct.stats.size[idx] = stats.size;
		deviceStruct.stats.maxEnergy[idx] = stats.maxEnergy;
		deviceStruct.stats.energyUsage[idx] = stats.energyUsage;
		deviceStruct.stats.sightDist[idx] = stats.sightDist;
		deviceStruct.stats.sightAngle[idx] = stats.sightAngle;
	}

	static Fish readFromDeviceStruct(const FishSoA& deviceStruct, int idx)
	{
		Stats stats;
		stats.energyUsage = deviceStruct.stats.energyUsage[idx];
		stats.maxEnergy = deviceStruct.stats.maxEnergy[idx];
		stats.sightAngle = deviceStruct.stats.sightAngle[idx];
		stats.sightDist = deviceStruct.stats.sightDist[idx];
		stats.size = deviceStruct.stats.size[idx];

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
			stats,
			deviceStruct.alives[idx],
			deviceStruct.nextDecisions[idx],
			deviceStruct.currentEnergy[idx]
		);
	}
private:
};
#endif // !FISH_CUH