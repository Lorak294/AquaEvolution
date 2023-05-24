#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "headers/Shader.h"
//#include "headers/aquarium.cuh"
#include "headers/structs/aquarium.cuh"
#include "headers/scene.cuh"
#include "headers/deviceFunctions.cuh"
#include "headers/simulation_kernels.cuh"

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <random>
#include <thread>
#include <chrono>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define checkCudaErrors(call)                                 \
  do {                                                        \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
      printf("CUDA error at %s %d: %s\n", __FILE__, __LINE__, \
             cudaGetErrorString(err));                        \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);
GLFWwindow* createGLWindow();
void createBuffers();
void createBackgroundBuffers();
void createBackgroundBuffers();
void createAlgaBuffers();
void createFishBuffers();
void cleanup();
void renderLoop(GLFWwindow* window, Shader shader);
void makeNewGeneration();
void initMemory();
void copyAquariumStructToDevice();
void copyAquariumStructFromDevice();
void freeHostAquariumStruct();
void freeDeviceAquariumStruct();
void freeDeviceSceneStruct();
void mallocHostAquariumStruct(int fishesCount, int algaeCount);
void mallocDeviceAquariumStruct(int fishesCount, int algaeCount);
void mallocDeviceSceneStruct();
void renderAquarium(Shader shader);
glm::mat4 getMVP(float2 pos, float2 vec, float size);

// openGL parameters
unsigned int SCR_WIDTH = 1000;
unsigned int SCR_HEIGHT = 1000;
const glm::vec3 SURFACE = { 0.0f, 0.9f, 0.9f };
const glm::vec3 DEEPWATER = { 0.0f, 0.0f, 0.0f };
const glm::vec3 ALGAECOLOR = { 0.0f, 1.0f, 0.0f };
const glm::vec3 FISHCOLOR1 = { 0.94f, 0.54f, 0.09f };
const glm::vec3 FISHCOLOR2 = { 0.85f, 0.7f, 0.2f };

// scene settings
//const float Object::initaialSize = 1.0f;
const unsigned int CELLSX = 1000;
const unsigned int CELLSY = 1000;
const unsigned int MAXOBJCOUNT = Aquarium::maxObjCount;
//const int Aquarium::maxObjCount = MAXOBJCOUNT;
unsigned int AQUARIUM_WIDTH = 100;
unsigned int AQUARIUM_HEIGHT = 100;

Aquarium hostAquarium;
AquariumSoA hostAquariumStruct;
AquariumSoA deviceAquariumStruct;
s_scene deviceSceneStruct;

unsigned int VBO_bg, VAO_bg, EBO_bg;
unsigned int VBO_alga, VAO_alga, EBO_alga;
unsigned int VBO_fish, VAO_fish, EBO_fish;

#include <curand_kernel.h>
curandState* randomGenerators;

__global__ void setup_random_generators(curandState* state)
{
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence
		   number, no offset */
	curand_init(2137, id, 0, &state[id]);
}


__global__ void test_random_generators(curandState* state)
{
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	float result = curand_uniform(&state[id]);
	printf("[%u] curand_uniform() = %f\n", id, result);
}

int main()
{
	srand(static_cast<unsigned> (time(0)));

	GLFWwindow* window = createGLWindow();

	// create shader
	Shader shader("texture.vs", "texture.fs");

	// iniitalize memory buffers
	initMemory();

	// initialize scene with radom objects
	//hostAquarium.radnomGeneration(10, 20, 0, AQUARIUM_WIDTH, 0, AQUARIUM_HEIGHT);
	hostAquarium.radnomGeneration(100, 400, 0, AQUARIUM_WIDTH, 0, AQUARIUM_HEIGHT);
	//hostAquarium.radnomGeneration(1000, 40000, -50, 150, -50, 150);
	//hostAquarium.radnomGeneration(1, 1, -50, 150, -50, 150);

	// creating openGL buffers and drawing
	createBuffers();
	renderLoop(window, shader);

	// cleanup
	cleanup();
	return 0;
}

// -------------------------------------------- OPENGL FUNCTIONS -----------------------------------------------
// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
void processInput(GLFWwindow *window)
{
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
	{
		glfwSetWindowShouldClose(window, true);
	}
}
// glfw: whenever the window size changed (by OS or user resize) this callback function executes
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	// make sure the viewport matches the new window dimensions; note that width and 
	// height will be significantly larger than specified on retina displays.
	glViewport(0, 0, width, height);

	// update parameters
	SCR_WIDTH = width;
	SCR_HEIGHT = height;
}
GLFWwindow* createGLWindow()
{
	// glfw: initialize and configure
	// ------------------------------
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	// glfw window creation
	// --------------------
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "AquaEvolution", NULL, NULL);
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		exit(-1);
	}
	glfwMakeContextCurrent(window);
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	//glfwSetKeyCallback(window, key_callback);

	// glad: load all OpenGL function pointers
	// ---------------------------------------
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize GLAD" << std::endl;
		exit(-1);
	}
	return window;
}
void createBuffers()
{
	createBackgroundBuffers();
	createAlgaBuffers();
	createFishBuffers();
}
void createBackgroundBuffers()
{
	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 1.0f,  1.0f, 1.0f, SURFACE.r,		SURFACE.g,		SURFACE.b ,		// top right
		 1.0f, -1.0f, 0.0f, DEEPWATER.r,	DEEPWATER.g,	DEEPWATER.b ,	// bottom right
		-1.0f, -1.0f, 0.0f, DEEPWATER.r,	DEEPWATER.g,	DEEPWATER.b ,	// bottom left
		-1.0f,  1.0f, 0.0f, SURFACE.r,		SURFACE.g,		SURFACE.b ,		// top left 
	};
	unsigned int indices[] =
	{
		0, 1, 3,   // first triangle
		1, 2, 3    // second triangle
	};


	glGenVertexArrays(1, &VAO_bg);
	glGenBuffers(1, &VBO_bg);
	glGenBuffers(1, &EBO_bg);

	glBindVertexArray(VAO_bg);

	glBindBuffer(GL_ARRAY_BUFFER, VBO_bg);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_bg);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}
void createAlgaBuffers()
{
	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 0.0f,  1.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top 
		 0.7f,  0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top right
		 1.0f,  0.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// left
		 0.7f, -0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom right 
		 0.0f, -1.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom
		-0.7f, -0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// bottom left 
		-1.0f,  0.0f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// left 
		-0.7f,  0.7f, 0.0f, ALGAECOLOR.r,	ALGAECOLOR.g,	ALGAECOLOR.b,	// top left 
	};
	unsigned int indices[] =
	{
		4, 5, 6,   
		4, 6, 7,   
		4, 7, 0,    
		4, 0, 1,    
		4, 1, 2,
		4, 2, 3
	};


	glGenVertexArrays(1, &VAO_alga);
	glGenBuffers(1, &VBO_alga);
	glGenBuffers(1, &EBO_alga);

	glBindVertexArray(VAO_alga);

	glBindBuffer(GL_ARRAY_BUFFER, VBO_alga);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_alga);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}
void createFishBuffers()
{
	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] =
	{	// coords			// colors 
		 0.0f,  1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// top 
		 0.3f,  0.0f, 0.0f, FISHCOLOR2.r,	FISHCOLOR2.g,	FISHCOLOR2.b,	// top right
		-0.3f,  0.0f, 0.0f, FISHCOLOR2.r,	FISHCOLOR2.g,	FISHCOLOR2.b,	// top left
		 0.0f, -0.5f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// bottom 
		-0.4f, -1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// tail left
		 0.4f, -1.0f, 0.0f, FISHCOLOR1.r,	FISHCOLOR1.g,	FISHCOLOR1.b,	// tail right 
	};
	unsigned int indices[] =
	{
		2, 1, 0,
		2, 3, 1,
		4, 5, 3
	};


	glGenVertexArrays(1, &VAO_fish);
	glGenBuffers(1, &VBO_fish);
	glGenBuffers(1, &EBO_fish);

	glBindVertexArray(VAO_fish);

	glBindBuffer(GL_ARRAY_BUFFER, VBO_fish);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO_fish);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// color attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}

// --------------------------------------------- MEMORY MANAGEMENT FUNCTIONS ------------------------------------
void cleanup()
{
	// de-allocate memory
	freeHostAquariumStruct();
	freeDeviceAquariumStruct();
	freeDeviceSceneStruct();

	// de-allocate all openGL buffers once they've outlived their purpose:
	glDeleteVertexArrays(1, &VAO_bg);
	glDeleteBuffers(1, &VBO_bg);
	glDeleteBuffers(1, &EBO_bg);

	glDeleteVertexArrays(1, &VAO_alga);
	glDeleteBuffers(1, &VBO_alga);
	glDeleteBuffers(1, &EBO_alga);

	glDeleteVertexArrays(1, &VAO_fish);
	glDeleteBuffers(1, &VBO_fish);
	glDeleteBuffers(1, &EBO_fish);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();
}
// --------------------------------------------- RENDERING FUNCTIONS ----------------------------------------------
void renderLoop(GLFWwindow* window, Shader shader)
{
	constexpr uint64_t THREADS_COUNT = 1024;
	constexpr uint64_t GENERATORS_BlOCK_COUNT = GENERATORS_COUNT / THREADS_COUNT + (GENERATORS_COUNT % THREADS_COUNT == (uint64_t)0 ? 0 : 1);

	// Initiate random float generators
	setup_random_generators <<<GENERATORS_BlOCK_COUNT, THREADS_COUNT>>> (randomGenerators);
	checkCudaErrors(cudaDeviceSynchronize());

	// we operate in simple 2D space so form MVP matrices M is enough
	while (!glfwWindowShouldClose(window))
	{
		// input
		processInput(window);

		// TEMPORARY - to see better
		std::this_thread::sleep_for(std::chrono::milliseconds(50));

		hostAquarium.writeToDeviceStruct(hostAquariumStruct);
		copyAquariumStructToDevice();

		int nblocks = std::max(
			hostAquariumStruct.fishes.count,
			hostAquariumStruct.algae.count) / THREADS_COUNT + 1;

		//auto start = std::chrono::high_resolution_clock::now();
		simulate_generation << <nblocks, THREADS_COUNT, 2 * sizeof(uint32_t) >> > (deviceAquariumStruct, randomGenerators);
		//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() << std::endl;

		checkCudaErrors(cudaDeviceSynchronize());
		copyAquariumStructFromDevice();
		hostAquarium.readFromDeviceStruct(hostAquariumStruct, true);

		// render scene
		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);
		renderAquarium(shader);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();
	}
}

// --------------------------------------------- AQUARIUM MANAGEMENT FUNCTIONS ----------------------------------------------
void makeNewGeneration()
{
	// copy alive objects to new aquarium
	hostAquarium.readFromDeviceStruct(hostAquariumStruct, false);

	// make new objects from each one
	hostAquarium.newGeneration();

	// apply mutations and store objects in new arrays
	hostAquarium.writeToDeviceStruct(hostAquariumStruct);
}
void renderAquarium(Shader shader)
{
	shader.use();

	// render background
	glBindVertexArray(VAO_bg);
	shader.setMat4("mvp", glm::mat4(1.0f));
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	// render objects
	for each (Fish o in hostAquarium.fishes)
	{
		if (o.is_alive == FishAliveEnum::DEAD) continue;

		shader.setMat4("mvp", getMVP(o.position,o.directionVec,o.size));

		// render fish
		glBindVertexArray(VAO_fish);
		glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
	}

	for each (Algae o in hostAquarium.algae)
	{
		if (!o.is_alive) continue;

		shader.setMat4("mvp", getMVP(o.position, o.directionVec, o.size));

		// render alga
		glBindVertexArray(VAO_alga);
		glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);
	}
}

glm::mat4 getMVP(float2 pos, float2 vec, float size)
{
	float scaleX = (2.0f / AQUARIUM_WIDTH);
	float scaleY = (2.0f / AQUARIUM_HEIGHT);
	glm::vec3 dirVec = glm::normalize(glm::vec3(vec.x, vec.y, 0));
	float angle = glm::acos(glm::dot(dirVec, glm::vec3(0, 1, 0)));
	if (vec.x > 0)
		angle *= -1.0f;
	glm::mat4 mvpMat = glm::mat4(1.0f);
	mvpMat = glm::scale(mvpMat, glm::vec3(scaleX, scaleY, 1));
	mvpMat = mvpMat = glm::translate(mvpMat, glm::vec3(pos.x - AQUARIUM_WIDTH / 2, pos.y - AQUARIUM_HEIGHT / 2, 0));
	mvpMat = glm::rotate(mvpMat, angle, glm::vec3(0.0, 0.0, 1.0));
	mvpMat = glm::scale(mvpMat, glm::vec3(size, size, 1));
	return mvpMat;
}

// --------------------------------------------- MEMORY MANAGEMENT FUNCTIONS ----------------------------------------------
void initMemory()
{
	mallocHostAquariumStruct(Aquarium::maxObjCount, Aquarium::maxObjCount);
	mallocDeviceAquariumStruct(Aquarium::maxObjCount, Aquarium::maxObjCount);
	mallocDeviceSceneStruct();
}
void copyAquariumStructToDevice()
{
	// Fish
	int nFish = deviceAquariumStruct.fishes.count = hostAquariumStruct.fishes.count;
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.positions.x, hostAquariumStruct.fishes.positions.x, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.positions.y, hostAquariumStruct.fishes.positions.y, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.directionVecs.x, hostAquariumStruct.fishes.directionVecs.x, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.directionVecs.y, hostAquariumStruct.fishes.directionVecs.y, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.alives, hostAquariumStruct.fishes.alives, sizeof(FishDecisionEnum)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.nextDecisions, hostAquariumStruct.fishes.nextDecisions, sizeof(FishDecisionEnum)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.sizes, hostAquariumStruct.fishes.sizes, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.hunger, hostAquariumStruct.fishes.hunger, sizeof(float)*nFish, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.fishes.interactionEntityIds, hostAquariumStruct.fishes.interactionEntityIds, sizeof(uint64_t)*nFish, cudaMemcpyHostToDevice));

	// Algae
	int nAlgae = deviceAquariumStruct.algae.count = hostAquariumStruct.algae.count;
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.positions.x, hostAquariumStruct.algae.positions.x, sizeof(float)*nAlgae, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.positions.y, hostAquariumStruct.algae.positions.y, sizeof(float)*nAlgae, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.directionVecs.x, hostAquariumStruct.algae.directionVecs.x, sizeof(float)*nAlgae, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.directionVecs.y, hostAquariumStruct.algae.directionVecs.y, sizeof(float)*nAlgae, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.alives, hostAquariumStruct.algae.alives, sizeof(bool)*nAlgae, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.algae.sizes, hostAquariumStruct.algae.sizes, sizeof(float)*nAlgae, cudaMemcpyHostToDevice));

}
void copyAquariumStructFromDevice()
{
	int nFish = hostAquariumStruct.fishes.count = deviceAquariumStruct.fishes.count;
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.positions.x, deviceAquariumStruct.fishes.positions.x, sizeof(float)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.positions.y, deviceAquariumStruct.fishes.positions.y, sizeof(float)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.directionVecs.x, deviceAquariumStruct.fishes.directionVecs.x, sizeof(float)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.directionVecs.y, deviceAquariumStruct.fishes.directionVecs.y, sizeof(float)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.alives, deviceAquariumStruct.fishes.alives, sizeof(FishAliveEnum)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.nextDecisions, deviceAquariumStruct.fishes.nextDecisions, sizeof(FishDecisionEnum)*nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.sizes, deviceAquariumStruct.fishes.sizes, sizeof(float) * nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.hunger, deviceAquariumStruct.fishes.hunger, sizeof(float) * nFish, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.fishes.interactionEntityIds, deviceAquariumStruct.fishes.interactionEntityIds, sizeof(uint64_t) * nFish, cudaMemcpyDeviceToHost));

	int nAlgae = hostAquariumStruct.algae.count = deviceAquariumStruct.algae.count;
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.positions.x, deviceAquariumStruct.algae.positions.x, sizeof(float)*nAlgae, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.positions.y, deviceAquariumStruct.algae.positions.y, sizeof(float)*nAlgae, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.directionVecs.x, deviceAquariumStruct.algae.directionVecs.x, sizeof(float)*nAlgae, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.directionVecs.y, deviceAquariumStruct.algae.directionVecs.y, sizeof(float)*nAlgae, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.alives, deviceAquariumStruct.algae.alives, sizeof(bool)*nAlgae, cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.algae.sizes, deviceAquariumStruct.algae.sizes, sizeof(float) * nAlgae, cudaMemcpyDeviceToHost));
}
void mallocHostAquariumStruct(int fishesCount, int algaeCount)
{
	hostAquariumStruct.fishes.count = fishesCount;
	hostAquariumStruct.fishes.positions.x = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.positions.y = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.directionVecs.x = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.directionVecs.y = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.alives = (FishAliveEnum*)malloc(fishesCount * sizeof(FishAliveEnum));
	hostAquariumStruct.fishes.nextDecisions = (FishDecisionEnum*)malloc(fishesCount * sizeof(bool));
	hostAquariumStruct.fishes.sizes = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.hunger = (float*)malloc(fishesCount * sizeof(float));
	hostAquariumStruct.fishes.interactionEntityIds = (uint64_t*)malloc(fishesCount * sizeof(float));

	hostAquariumStruct.algae.count = algaeCount;
	hostAquariumStruct.algae.positions.x = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.positions.y = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.directionVecs.x = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.directionVecs.y = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.alives = (bool*)malloc(algaeCount * sizeof(bool));
	hostAquariumStruct.algae.sizes = (float*)malloc(algaeCount * sizeof(float));
}
void mallocDeviceAquariumStruct(int fishesCount, int algaeCount)
{
	// Fish
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.positions.x, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.positions.y, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.directionVecs.x, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.directionVecs.y, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.alives, sizeof(FishAliveEnum)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.nextDecisions, sizeof(FishDecisionEnum)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.sizes, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.hunger, sizeof(float)*fishesCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.fishes.interactionEntityIds, sizeof(uint64_t)*fishesCount));

	// Algae
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.positions.x, sizeof(float)*algaeCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.positions.y, sizeof(float)*algaeCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.directionVecs.x, sizeof(float)*algaeCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.directionVecs.y, sizeof(float)*algaeCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.alives, sizeof(bool)*algaeCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.algae.sizes, sizeof(float)*algaeCount));

	// Random generators
	checkCudaErrors(cudaMalloc((void**)&randomGenerators, sizeof(curandState) * GENERATORS_COUNT));
}
void mallocDeviceSceneStruct()
{
	deviceSceneStruct.cellX = CELLSX;
	deviceSceneStruct.cellY = CELLSY;
	deviceSceneStruct.cellHieght = AQUARIUM_HEIGHT / CELLSY;
	deviceSceneStruct.cellWidth = AQUARIUM_WIDTH / CELLSX;
	checkCudaErrors(cudaMalloc((void**)&deviceSceneStruct.objCellArray, MAXOBJCOUNT *sizeof(uint)));
	checkCudaErrors(cudaMalloc((void**)&deviceSceneStruct.cellSizeArray, CELLSX*CELLSY*2*sizeof(uint)));
}
void freeDeviceAquariumStruct()
{
	// Fish
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.positions.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.positions.y));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.directionVecs.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.directionVecs.y));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.alives));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.nextDecisions));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.sizes));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.hunger));
	checkCudaErrors(cudaFree(deviceAquariumStruct.fishes.interactionEntityIds));

	// Algae
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.positions.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.positions.y));
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.directionVecs.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.directionVecs.y));
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.alives));
	checkCudaErrors(cudaFree(deviceAquariumStruct.algae.sizes));

	// Random generators
	checkCudaErrors(cudaFree(randomGenerators));
}
void freeHostAquariumStruct()
{
	free(hostAquariumStruct.fishes.positions.x);
	free(hostAquariumStruct.fishes.positions.y);

	free(hostAquariumStruct.fishes.directionVecs.x);
	free(hostAquariumStruct.fishes.directionVecs.y);

	free(hostAquariumStruct.fishes.alives);
	free(hostAquariumStruct.fishes.nextDecisions);
	free(hostAquariumStruct.fishes.sizes);
	free(hostAquariumStruct.fishes.hunger);
	free(hostAquariumStruct.fishes.interactionEntityIds);

	free(hostAquariumStruct.algae.positions.x);
	free(hostAquariumStruct.algae.positions.y);

	free(hostAquariumStruct.algae.directionVecs.x);
	free(hostAquariumStruct.algae.directionVecs.y);

	free(hostAquariumStruct.algae.alives);
	free(hostAquariumStruct.algae.sizes);
}
void freeDeviceSceneStruct()
{
	checkCudaErrors(cudaFree(deviceSceneStruct.objCellArray));
	checkCudaErrors(cudaFree(deviceSceneStruct.cellSizeArray));
}