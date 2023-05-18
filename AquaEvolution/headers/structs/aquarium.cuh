#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include "fish.cuh"
#include "algae.cuh"
#include <vector>


template<class T> static std::vector<T> createChildren(const T& parent);
static float2 randomVector();
static float randomFloat(float a, float b);

struct AquariumSoA
{
	FishSoA fishes;
	AlgaeSoA algae;
};

class Aquarium
{
public:
	static constexpr uint64_t maxObjCount = 10000;
	std::vector<Fish> fishes;
	std::vector<Algae> algae;
	static constexpr float WIDTH = 100.0f;
	static constexpr float HEIGHT = 100.0f;

	Aquarium() {}

	void readFromDeviceStruct(const AquariumSoA& deviceStruct, bool includeDead)
	{
		fishes.clear();
		for (int i = 0; i < deviceStruct.fishes.count; i++)
		{
			if (includeDead || deviceStruct.fishes.alives[i])
			{
				fishes.push_back(Fish::readFromDeviceStruct(deviceStruct.fishes, i));
			}
		}

		algae.clear();
		for (int i = 0; i < deviceStruct.algae.count; i++)
		{
			if (includeDead || deviceStruct.algae.alives[i])
			{
				algae.push_back(Algae::readFromDeviceStruct(deviceStruct.algae, i));
			}
		}
	}

	void writeToDeviceStruct(AquariumSoA& deviceStruct)
	{
		// NOTE: assuming deviceStruct has already been reallocated to new size
		int copySize = fishes.size() > maxObjCount ? maxObjCount : fishes.size();

		deviceStruct.fishes.count = copySize;
		for (int i = 0; i < copySize; i++)
		{
			fishes[i].writeToDeviceStruct(deviceStruct.fishes, i);
		}

		// NOTE: assuming deviceStruct has already been reallocated to new size
		copySize = algae.size() > maxObjCount ? maxObjCount : algae.size();

		deviceStruct.algae.count = copySize;
		for (int i = 0; i < copySize; i++)
		{
			algae[i].writeToDeviceStruct(deviceStruct.algae, i);
		}
	}

	// generates new generation based on current arrays state and stores data in arrays
	void newGeneration()
	{
		int steps = fishes.size();
		for (int i = 0; i < steps; i++)
		{
			// grow curr object
			fishes[i].size *= 1.5f;

			// create new generation
			std::vector<Fish> children = createChildren(fishes[i]);
			fishes.insert(fishes.end(), children.begin(), children.end());
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
		fishes.clear();
		for (int i = 0; i < fishCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			fishes.push_back(Fish(pos, randomVector(), Fish::initaialSize, true));
		}

		algae.clear();
		for (int i = 0; i < algaeCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			algae.push_back(Algae(pos, randomVector(), Algae::initaialSize, true));
		}
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

#endif // !AQUARIUM_CUH
