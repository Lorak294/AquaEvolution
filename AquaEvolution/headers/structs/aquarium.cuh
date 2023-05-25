#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include "fish.cuh"
#include "algae.cuh"
#include <vector>
#include <cassert>


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
	std::vector<Fish> fishes_second;
	std::vector<Algae> algae;
	std::vector<Algae> algae_second;
	static constexpr float WIDTH = 100.0f;
	static constexpr float HEIGHT = 100.0f;

	Aquarium() 
	{
		fishes.reserve(maxObjCount);
		fishes_second.reserve(maxObjCount);
		algae.reserve(maxObjCount);
		algae_second.reserve(maxObjCount);
	}

	void readFromDeviceStruct(const AquariumSoA& deviceStruct, bool includeDead)
	{
		fishes.clear();
		for (int i = 0; i < *deviceStruct.fishes.count; i++)
		{
			if (includeDead || deviceStruct.fishes.alives[i] == FishAliveEnum::ALIVE)
			{
				fishes.push_back(Fish::readFromDeviceStruct(deviceStruct.fishes, i));
			}
		}

		algae.clear();
		for (int i = 0; i < *deviceStruct.algae.count; i++)
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

		*deviceStruct.fishes.count = copySize;
		for (int i = 0; i < copySize; i++)
		{
			fishes[i].writeToDeviceStruct(deviceStruct.fishes, i);
		}

		// NOTE: assuming deviceStruct has already been reallocated to new size
		copySize = algae.size() > maxObjCount ? maxObjCount : algae.size();

		*deviceStruct.algae.count = copySize;
		for (int i = 0; i < copySize; i++)
		{
			algae[i].writeToDeviceStruct(deviceStruct.algae, i);
		}
	}

	// generates new generation based on current arrays state and stores data in arrays
	void newGeneration()
	{
		int steps = fishes.size();
		fishes_second.clear();
		for (int i = 0; i < steps; i++)
		{
			if (fishes[i].is_alive == FishAliveEnum::DEAD) continue;

			// grow curr object
			fishes[i].size *= 1.1f;
			fishes_second.push_back(fishes[i]);

			// create new generation
			std::vector<Fish> children = createChildren(fishes[i]);
			assert((void
				("fishes size + children >= fishes capacity"),
				(fishes_second.size() + children.size() < fishes_second.capacity())));

			fishes_second.insert(fishes_second.end(), children.begin(), children.end());
		}
		fishes = fishes_second;

		//algae_second.clear();
		//steps = algae.size();
		//for (int i = 0; i < steps; i++)
		//{
		//	if (!algae[i].is_alive) continue;

		//	// grow curr object
		//	algae[i].size *= 1.1f;
		//	algae_second.push_back(algae[i]);

		//	// create new generation
		//	std::vector<Algae> children = createChildren(algae[i]);
		//	assert((void
		//			("algae size + children >= algae capacity"),
		//			algae_second.size() + children.size() < algae_second.capacity()));

		//	algae_second.insert(algae_second.end(), children.begin(), children.end());
		//}
		//algae = algae_second;
	}

	// generates new generation based of given number of objects with random values
	void radnomGeneration(int fishCount, int algaeCount, int Xbeg, int Xend, int Ybeg, int Yend)
	{
		fishes.clear();
		for (int i = 0; i < fishCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			fishes.push_back(Fish(pos, randomVector(), Fish::initaialSize));
		}

		algae.clear();
		for (int i = 0; i < algaeCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			algae.push_back(Algae(pos, randomVector(), Algae::initaialSize));
		}
	}
};

template<class T>
std::vector<T> createChildren(const T& parent)
{
	// TEMPORARY IMPEMENTATION
	std::vector<T> children;
	float offsetx = randomFloat(1.0f, 2.0f);
	float offsety = randomFloat(1.0f, 2.0f);

	children.push_back(T(
		{ 
			parent.position.x + offsetx,
			parent.position.y + offsety,
		}, 
		randomVector(), 
		//T::initaialSize));
		parent.size));
	return children;
}

static float2 randomVector()
{
	float2 vec = { randomFloat(-1, 1), randomFloat(-1, 1) };
	return normalize(vec);
}

static float randomFloat(float a, float b)
{
	return a + (b - a)*(float)(rand()) / (float)(RAND_MAX);
}

#endif // !AQUARIUM_CUH
