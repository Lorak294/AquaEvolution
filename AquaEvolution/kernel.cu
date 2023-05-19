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
#include <chrono>
#include <random>

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
void makeNewGeneration(AquariumThrustContainer& thrustAquarium);
void renderAquarium(Shader shader);
void calcIndexesFromSizes(const std::vector<uint>& sizes, std::vector<uint>& idxArray);
glm::mat4 getMVP(float2 pos, float2 vec);
void prepareAndSortScene(AquariumThrustContainer& thrustAquarium, SceneThrustContainer& thrustScene);
void fillKernelStructs(AquariumThrustContainer& thrustAquarium, SceneThrustContainer& thrustScene,
	AquariumSoA& kernelAquarium, SceneSoA& kernelScene);

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
const unsigned int MAXOBJCOUNT = 100000;
//const int Aquarium::maxObjCount = MAXOBJCOUNT;
unsigned int AQUARIUM_WIDTH = 100;
unsigned int AQUARIUM_HEIGHT = 100;

// global variables 
Aquarium hostAquarium;

unsigned int VBO_bg, VAO_bg, EBO_bg;
unsigned int VBO_alga, VAO_alga, EBO_alga;
unsigned int VBO_fish, VAO_fish, EBO_fish;

int main()
{
	srand(static_cast<unsigned> (time(0)));

	GLFWwindow* window = createGLWindow();

	// create shader
	Shader shader("texture.vs", "texture.fs");

	// iniitalize memory buffers

	// initialize scene with radom objects
	hostAquarium.radnomGeneration(10, 20, 0, AQUARIUM_WIDTH, 0, AQUARIUM_HEIGHT);
	//hostAquarium.radnomGeneration(1, 1, 0, AQUARIUM_WIDTH, 0, AQUARIUM_HEIGHT);

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
	// new every generation
	SceneThrustContainer thrustScene;
	AquariumThrustContainer thrustAquarium;

	// we operate in simple 2D space so form MVP matrices M is enough
	while (!glfwWindowShouldClose(window))
	{
		// input
		processInput(window);

		// KERNEL HERE
		auto start = std::chrono::high_resolution_clock::now();
		hostAquarium.writeToDevice(thrustAquarium);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Writing to device: "<< duration.count() << " microseconds" << std::endl;

		// prepare structs to pass to kernel (objects with sorting by cell position)
		prepareAndSortScene(thrustAquarium, thrustScene);
		AquariumSoA kernelAquarium;
		SceneSoA kernelScene;
		fillKernelStructs(thrustAquarium, thrustScene, kernelAquarium, kernelScene);

		const int n = 1024;
		int nblocks = std::max(
			thrustAquarium.fish.alives.size(),
			thrustAquarium.algae.alives.size()) / n + 1;
		//simulateGeneration <<<nblocks, n>>> (deviceAquariumStruct, deviceSceneStruct);
		start = std::chrono::high_resolution_clock::now();
		decision_fish << <nblocks, n >> > (kernelAquarium, kernelScene);
		decision_algae << <nblocks, n >> > (kernelAquarium, kernelScene);
		checkCudaErrors(cudaDeviceSynchronize());

		move_fish << <nblocks, n >> > (kernelAquarium, kernelScene);
		move_algae << <nblocks, n >> > (kernelAquarium, kernelScene);
		checkCudaErrors(cudaDeviceSynchronize());
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Kernels execution: " << duration.count() << " microseconds" << std::endl;

		start = std::chrono::high_resolution_clock::now();
		hostAquarium.readFromDevice(thrustAquarium, true);
		stop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
		std::cout << "Reading from device: " << duration.count() << " microseconds" << std::endl;

		//// Decision phase
		//decision_fish();
		//decision_algae();
		//cudaDeviceSynchronize();

		//// move phase
		//move_fish();
		//move_algae();
		//cudaDeviceSynchronize();


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
void prepareAndSortScene(AquariumThrustContainer& thrustAquarium, SceneThrustContainer& thrustScene)
{
	float cellWidth = AQUARIUM_WIDTH / CELLSX;
	float cellHeight = AQUARIUM_WIDTH / CELLSY;
	
	// local host-side arrays of cellSizes
	std::vector<uint> fishCellSizes(CELLSX*CELLSY,0);
	std::vector<uint> algaeCellSizes(CELLSX*CELLSY,0);

	// calculating device arrays for sorting and filling local host-side sizes
	auto start = std::chrono::high_resolution_clock::now();
	thrustScene.fishCellPositioning = hostAquarium.calcFishCellArray(cellWidth, cellHeight,CELLSX,fishCellSizes);
	thrustScene.algaeCellPositioning = hostAquarium.calcFishCellArray(cellWidth, cellHeight,CELLSX,algaeCellSizes);
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "filling cellPositioning arrays: " << duration.count() << " microseconds" << std::endl;
	
	// calculate device idxArrays
	start = std::chrono::high_resolution_clock::now();
	std::vector<uint> cellIndexesFish(CELLSX*CELLSY, 0);
	std::vector<uint> cellIndexesAlgae(CELLSX*CELLSY, 0);
	calcIndexesFromSizes(fishCellSizes, cellIndexesFish);
	calcIndexesFromSizes(algaeCellSizes, cellIndexesAlgae);
	thrustScene.cellIndexesFish = cellIndexesFish;
	thrustScene.cellIndexesAlgae = cellIndexesAlgae;
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "calculating Idx arrays: " << duration.count() << " microseconds" << std::endl;
	
	// copy sizes to device
	thrustScene.cellSizesFish = fishCellSizes;
	thrustScene.cellSizesAlgae = algaeCellSizes;

	start = std::chrono::high_resolution_clock::now();
	thrust::sort_by_key(thrustScene.fishCellPositioning.begin(), thrustScene.fishCellPositioning.end(),
		thrust::make_zip_iterator(make_tuple(
			thrustAquarium.fish.alives.begin(),
			thrustAquarium.fish.directionVecs.begin(),
			thrustAquarium.fish.positions.begin(),
			thrustAquarium.fish.sizes.begin())));
	thrust::sort_by_key(thrustScene.algaeCellPositioning.begin(), thrustScene.algaeCellPositioning.end(),
		thrust::make_zip_iterator(make_tuple(
			thrustAquarium.algae.alives.begin(),
			thrustAquarium.algae.directionVecs.begin(),
			thrustAquarium.algae.positions.begin(),
			thrustAquarium.algae.sizes.begin())));
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	std::cout << "ZIP iterator sorting: " << duration.count() << " microseconds" << std::endl;
	// sort positioning by hash together with arguments
}

void fillKernelStructs(AquariumThrustContainer& thrustAquarium, SceneThrustContainer& thrustScene,
	AquariumSoA& kernelAquarium, SceneSoA& kernelScene)
{
	// algae
	kernelAquarium.algae.count = thrustAquarium.algae.alives.size();
	kernelAquarium.algae.alives = thrust::raw_pointer_cast(&thrustAquarium.algae.alives[0]);
	kernelAquarium.algae.positions = thrust::raw_pointer_cast(&thrustAquarium.algae.positions[0]);
	kernelAquarium.algae.directionVecs = thrust::raw_pointer_cast(&thrustAquarium.algae.directionVecs[0]);
	kernelAquarium.algae.sizes = thrust::raw_pointer_cast(&thrustAquarium.algae.sizes[0]);

	kernelScene.algaeCellPositioning = thrust::raw_pointer_cast(&thrustScene.algaeCellPositioning[0]);
	kernelScene.cellSizesAlgae = thrust::raw_pointer_cast(&thrustScene.cellSizesAlgae[0]);
	kernelScene.cellIndexesAlgae = thrust::raw_pointer_cast(&thrustScene.cellIndexesAlgae[0]);

	// fish
	kernelAquarium.fish.count = thrustAquarium.fish.alives.size();
	kernelAquarium.fish.alives = thrust::raw_pointer_cast(&thrustAquarium.fish.alives[0]);
	kernelAquarium.fish.positions = thrust::raw_pointer_cast(&thrustAquarium.fish.positions[0]);
	kernelAquarium.fish.directionVecs = thrust::raw_pointer_cast(&thrustAquarium.fish.directionVecs[0]);
	kernelAquarium.fish.sizes = thrust::raw_pointer_cast(&thrustAquarium.fish.sizes[0]);
	kernelAquarium.fish.nextDecisions = thrust::raw_pointer_cast(&thrustAquarium.fish.nextDecisions[0]);
	kernelAquarium.fish.interactionEntityIds = thrust::raw_pointer_cast(&thrustAquarium.fish.interactionEntityIds[0]);

	kernelScene.fishCellPositioning = thrust::raw_pointer_cast(&thrustScene.fishCellPositioning[0]);
	kernelScene.cellSizesFish = thrust::raw_pointer_cast(&thrustScene.cellSizesFish[0]);
	kernelScene.cellIndexesFish = thrust::raw_pointer_cast(&thrustScene.cellIndexesFish[0]);
}

void calcIndexesFromSizes(const std::vector<uint>& sizes, std::vector<uint>& idxArray)
{
	idxArray[0] = 0;
	for (int i = 1; i < sizes.size(); i++)
	{
		idxArray[i] = idxArray[i - 1] + sizes[i - 1];
	}
}


void makeNewGeneration(AquariumThrustContainer& thrustAquarium)
{
	// copy alive objects to new aquarium
	hostAquarium.readFromDevice(thrustAquarium, false);

	// make new objects from each one
	hostAquarium.newGeneration();

	// apply mutations and store objects in new arrays
	hostAquarium.writeToDevice(thrustAquarium);
}
void renderAquarium(Shader shader)
{
	shader.use();

	// render background
	glBindVertexArray(VAO_bg);
	shader.setMat4("mvp", glm::mat4(1.0f));
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	// render objects
	for each (Fish o in hostAquarium.fish)
	{
		shader.setMat4("mvp", getMVP(o.position,o.directionVec));

		// render fish
		glBindVertexArray(VAO_fish);
		glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
	}

	for each (Algae o in hostAquarium.algae)
	{
		if (!o.is_alive) continue;

		shader.setMat4("mvp", getMVP(o.position, o.directionVec));

		// render alga
		glBindVertexArray(VAO_alga);
		glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);
	}
}

glm::mat4 getMVP(float2 pos, float2 vec)
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
	return mvpMat;
}
