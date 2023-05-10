#ifndef AQUARIUM
#define AQUARIUM

#include "objects.cuh";
#include <vector>

float2 randomVector();
float randomFloat(float a, float b);
std::vector<Object> createChildren(const Object& parent);


struct s_aquarium 
{
	s_objects objects;
	int objectsCount;
};

class Aquarium
{
private:
	static const int maxObjCount;

public:
	std::vector<Object> objects;

	Aquarium(){}

	void readFromDeviceStruct(const s_aquarium& deviceStruct, bool includeDead)
	{
		objects.clear();
		for (int i = 0; i < deviceStruct.objectsCount; i++)
		{
			if (includeDead || deviceStruct.objects.alives[i])
			{
				objects.push_back(Object::readFromDeviceStruct(deviceStruct.objects, i));
			}
		}
	}

	void writeToDeviceStruct(s_aquarium& deviceStruct)
	{
		// NOTE: assuming deviceStruct has already been reallocated to new size
		int copySize = objects.size() > maxObjCount ? maxObjCount : objects.size();
		
		for (int i=0; i< copySize; i++)
		{
			objects[i].writeToDeviceStruct(deviceStruct.objects, i);
		}
	}

	// generates new generation based on current arrays state and stores data in arrays
	void newGeneration()
	{
		int steps = objects.size();
		for (int i = 0; i < steps; i++)
		{
			// grow curr object
			objects[i].size *= 1.5f;

			// create new generation
			std::vector<Object> children = createChildren(objects[i]);
			objects.insert(objects.end(), children.begin(), children.end());
		}
	}
	
	// generates new generation based of given number of objects with random values
	void radnomGeneration(int fishCount, int algaeCount, int Xbeg, int Xend, int Ybeg, int Yend)
	{
		objects.clear();
		for (int i = 0; i < fishCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			objects.push_back(Object(pos, randomVector(), Object::initaialSize, true, true));
		}
		for (int i = 0; i < algaeCount; i++)
		{
			float2 pos = { randomFloat(Xbeg,Xend), randomFloat(Ybeg,Yend) };
			objects.push_back(Object(pos, randomVector(), Object::initaialSize, false, true));
		}
	}

	thrust::host_vector<uint> createCellArray(float cellWidth, float cellHeight)
	{
		int copySize = objects.size() > maxObjCount ? maxObjCount : objects.size();
		
		thrust::host_vector<uint> cellArr(copySize);

		for (int i = 0; i < copySize; i++)
		{
			cellArr[i] = objects[i].hashCell(cellWidth, cellHeight);
		}
		return cellArr;
	}
};


std::vector<Object> createChildren(const Object& parent)
{
	// TEMPORARY IMPEMENTATION
	std::vector<Object> children;
	children.push_back(Object(parent.position, randomVector(), Object::initaialSize, parent.fish));
	return children;
}

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

