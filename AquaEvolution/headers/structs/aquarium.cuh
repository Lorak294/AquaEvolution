#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include "fish.cuh"
#include "algae.cuh"
#include <vector>

struct AquariumSoA
{
	FishSoA fish;
	AlgaeSoA algae;
};

struct AquariumThrustContainer
{
	FishThrustContainer fish;
	AlgaeThrustContainer algae;
};

template<class T> static std::vector<T> createChildren(const T& parent);
static float2 randomVector();
static float randomFloat(float a, float b);
static void resetThrustAquarium(AquariumThrustContainer& thrustAquarium, int algaeCount, int fishCount);

class Aquarium
{
public:
	static constexpr uint64_t maxFishCount = 10000;
	static constexpr uint64_t maxAlgaeCount = 10000;
	std::vector<Fish> fish;
	std::vector<Algae> algae;

	Aquarium() {}

	void readFromDevice(const AquariumThrustContainer& thrustAquarium, bool includeDead)
	{
		fish.clear();
		for (int i = 0; i < thrustAquarium.fish.alives.size(); i++)
		{
			if (includeDead || thrustAquarium.fish.alives[i])
			{
				fish.push_back(Fish::readFromDevice(thrustAquarium.fish, i));
			}
		}

		algae.clear();
		for (int i = 0; i < thrustAquarium.algae.alives.size(); i++)
		{
			if (includeDead || thrustAquarium.algae.alives[i])
			{
				algae.push_back(Algae::readFromDevice(thrustAquarium.algae, i));
			}
		}
	}

	void writeToDevice(AquariumThrustContainer& thrustAquarium)
	{
		int fishCopySize = fish.size() > maxFishCount ? maxFishCount : fish.size();
		int algaeCopySize = algae.size() > maxAlgaeCount ? maxAlgaeCount : algae.size();
		resetThrustAquarium(thrustAquarium, algaeCopySize, fishCopySize);

		for (int i = 0; i < fishCopySize; i++)
		{
			fish[i].writeToDevice(thrustAquarium.fish, i);
		}

		for (int i = 0; i < algaeCopySize; i++)
		{
			algae[i].writeToDevice(thrustAquarium.algae, i);
		}
	}

	// generates new generation based on current arrays state and stores data in arrays
	void newGeneration()
	{
		int steps = fish.size();
		for (int i = 0; i < steps; i++)
		{
			// grow curr object
			fish[i].size *= 1.5f;

			// create new generation
			std::vector<Fish> children = createChildren(fish[i]);
			fish.insert(fish.end(), children.begin(), children.end());
		}

		steps = algae.size();
		for (int i = 0; i < steps; i++)
		{
			// grow curr object
			algae[i].size *= 1.5f;

			// create new generation
			std::vector<Algae> children = createChildren(algae[i]);
			algae.insert(algae.end(), children.begin(), children.end());
		}
	}

	// generates new generation based of given number of objects with random values
	void radnomGeneration(int fishCount, int algaeCount, int Xbeg, int Xend, int Ybeg, int Yend)
	{
		fish.clear();
		for (int i = 0; i < fishCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			fish.push_back(Fish(pos, randomVector(), Fish::initaialSize, true));
		}

		algae.clear();
		for (int i = 0; i < algaeCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			algae.push_back(Algae(pos, randomVector(), Algae::initaialSize, true));
		}
	}

	// generates cell hash array of fish and fills cellSize array with counters 
	std::vector<uint> calcFishCellArray(float cellWidth, float cellHeight, int cellX, std::vector<uint>& cellSizeArray)
	{
		std::vector<uint> cellArray;

		for each (Fish f in fish)
		{
			// increment counter in size array
			cellSizeArray[f.calcCellIdx(cellWidth, cellHeight, cellX)]++;
			// add hash to positioning array
			cellArray.push_back(f.calcCellHash(cellWidth, cellHeight));
		}
		return cellArray;
	}

	// generates cell hash array of algae and fills cellSize array with counters 
	std::vector<uint> calcAlgaeCellArray(float cellWidth, float cellHeight, int cellX, std::vector<uint>& cellSizeArray)
	{
		std::vector<uint> cellArray;

		for each (Algae alga in algae)
		{
			// increment counter in size array
			cellSizeArray[alga.calcCellIdx(cellWidth, cellHeight, cellX)]++;
			// add hash to positioning array
			cellArray.push_back(alga.calcCellHash(cellWidth, cellHeight));
		}
		return cellArray;
	}
};

template<class T>
static std::vector<T> createChildren(const T& parent)
{
	// TEMPORARY IMPEMENTATION
	std::vector<T> children;
	children.push_back(T(parent.position, randomVector(), T::initaialSize));
	return children;
}

static float2 randomVector()
{
	float2 vec = { randomFloat(-1,1),randomFloat(-1,1) };
	return normalize(vec);
}

static float randomFloat(float a, float b)
{
	return a + (b - a)*(float)(rand()) / (float)(RAND_MAX);
}

static void resetThrustAquarium(AquariumThrustContainer& thrustAquarium,int algaeCount, int fishCount)
{
	thrustAquarium.algae.alives.resize(algaeCount);
	thrustAquarium.algae.positions.resize(algaeCount);
	thrustAquarium.algae.directionVecs.resize(algaeCount);
	thrustAquarium.algae.sizes.resize(algaeCount);

	thrustAquarium.fish.alives.resize(fishCount);
	thrustAquarium.fish.positions.resize(fishCount);
	thrustAquarium.fish.directionVecs.resize(fishCount);
	thrustAquarium.fish.sizes.resize(fishCount);
	thrustAquarium.fish.nextDecisions.resize(fishCount);
	thrustAquarium.fish.interactionEntityIds.resize(fishCount);

	//others...
}
#endif // !AQUARIUM_CUH