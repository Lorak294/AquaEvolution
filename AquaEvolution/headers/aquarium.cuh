#ifndef AQUARIUM
#define AQUARIUM

#include "algae.cuh";
#include <vector>

float2 randomVector();
float randomFloat(float a, float b);


struct s_aquarium 
{
	s_algae algae;
	int algaeCount;
};

class aquarium
{
public:
	std::vector<alga> algae;

	aquarium(){}

	void readFromDeviceStruct(const s_aquarium& deviceStruct, bool includeDead)
	{
		algae.clear();
		for (int i = 0; i < deviceStruct.algaeCount; i++)
		{
			if (includeDead || deviceStruct.algae.alives[i])
			{
				algae.push_back(alga::readFromDeviceStruct(deviceStruct.algae, i));
			}
		}
	}

	void writeToDeviceStruct(s_aquarium& deviceStruct)
	{
		// NOTE: assuming deviceStruct has already been reallocated to new size
		
		for (int i=0; i<algae.size(); i++)
		{
			algae[i].writeToDeviceStruct(deviceStruct.algae, i);
		}
	}

	// generates new generation based on current arrays state and stores data in arrays
	void newGeneration()
	{
		//auto end = algae.end();
		//for (auto it = algae.begin(); it != end; it++)
		//{
		//	// grow curr alaga
		//	it->size = it->size * 1.5f;

		//	// create new algae 
		//	alga childAlga(it->position, randomVector(), alga::initaialSize);
		//	algae.push_back(childAlga);
		//}

		int steps = algae.size();
		for (int i = 0; i < steps; i++)
		{
			// grow curr alaga
			algae[i].size *= 1.5f;

			// create new algae 
			alga childAlga(algae[i].position, randomVector(), alga::initaialSize);
			algae.push_back(childAlga);
		}
	}

};


float2 randomVector()
{
	float2 vec = { randomFloat(-1,1),randomFloat(-1,1) };
	return normalize(vec);
}

float randomFloat(float a, float b)
{
	return a + (b - a)*(float)(rand()) / (float)(RAND_MAX);
}

#endif // !AQUARIUM

