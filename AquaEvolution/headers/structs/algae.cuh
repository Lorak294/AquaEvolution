#include "../structures.cuh"
#ifndef ALGAE_CUH
#define ALGAE_CUH

struct AlgaeStatsSoA
{
	float* size;
	float* maxEnergy;
	float* energyUsage;
};

struct AlgaeSoA
{
	uint64_t* count;

	s_float2 positions;
	s_float2 directionVecs;
	bool* alives;
	float* currentEnergy;

	// ... other traits
	AlgaeStatsSoA stats;
};


struct AlgaeStats
{
	float size;
	float maxEnergy;
	float energyUsage;

	AlgaeStats() : size(0.1f), maxEnergy(50.0f), energyUsage(0.001f)
	{}
};

class Algae
{
	// FIELDS
public:
	static constexpr float initaialSize = 0.1f;
	static constexpr float initialEnergy = 25.0f;

	float2 position;
	float2 directionVec;
	bool is_alive;
	float currentEnergy;

	AlgaeStats stats;

	Algae(float2 p, float2 dv, AlgaeStats st, bool alive = true, float energy = initialEnergy)
		: position(p), directionVec(dv), is_alive(alive), currentEnergy(energy)
	{}

	// MEMBER FUNCTIONS
public:

	void writeToDeviceStruct(AlgaeSoA& deviceStruct, int idx)
	{
		deviceStruct.positions.x[idx] = position.x;
		deviceStruct.positions.y[idx] = position.y;
		deviceStruct.directionVecs.x[idx] = directionVec.x;
		deviceStruct.directionVecs.y[idx] = directionVec.y;
		deviceStruct.alives[idx] = is_alive;
		deviceStruct.currentEnergy[idx] = currentEnergy;

		deviceStruct.stats.size[idx] = stats.size;
		deviceStruct.stats.maxEnergy[idx] = stats.maxEnergy;
		deviceStruct.stats.energyUsage[idx] = stats.energyUsage;
	}

	static Algae readFromDeviceStruct(const AlgaeSoA& deviceStruct, int idx)
	{
		AlgaeStats stats;
		stats.size = deviceStruct.stats.size[idx];
		stats.maxEnergy = deviceStruct.stats.maxEnergy[idx];
		stats.energyUsage = deviceStruct.stats.energyUsage[idx];
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
			stats,
			deviceStruct.alives[idx],
			deviceStruct.currentEnergy[idx]
		);
	}
private:
};

#endif // !ALGAE_CUH
