#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "headers/Shader.h"
#include "headers/aquarium.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
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
void makeNewGeneration();
void initMemory();
void copyAquariumStructToDevice();
void copyAquariumStructFromDevice();
void freeHostAquariumStruct();
void freeDeviceAquariumStruct();
void freeDeviceSceneStruct();
void mallocHostAquariumStruct(int algaeCount);
void mallocDeviceAquariumStruct(int algaeCount);
void mallocDeviceSceneStruct();
void renderAquarium(Shader shader);

// openGL parameters
unsigned int SCR_WIDTH = 1000;
unsigned int SCR_HEIGHT = 1000;
const glm::vec3 SURFACE = { 0.0f, 0.9f, 0.9f };
const glm::vec3 DEEPWATER = { 0.0f, 0.0f, 0.0f };
const glm::vec3 ALGAECOLOR = { 0.0f, 1.0f, 0.0f };
const glm::vec3 FISHCOLOR1 = { 0.94f, 0.54f, 0.09f };
const glm::vec3 FISHCOLOR2 = { 0.85f, 0.7f, 0.2f };

// scene settings
const float Object::initaialSize = 1.0f;
const unsigned int CELLSX = 1000;
const unsigned int CELLSY = 1000;
const unsigned int MAXOBJCOUNT = 100000;
const int Aquarium::maxObjCount = MAXOBJCOUNT;
unsigned int AQUARIUM_WIDTH = 100;
unsigned int AQUARIUM_HEIGHT = 100;

// global variables 
Aquarium hostAquarium;
s_aquarium hostAquariumStruct;
s_aquarium deviceAquariumStruct;
s_scene deviceSceneStruct;

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
	initMemory();

	// initialize scene with radom objects
	hostAquarium.radnomGeneration(10, 20, 0, AQUARIUM_WIDTH, 0, AQUARIUM_HEIGHT);

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
	// we operate in simple 2D space so form MVP matrices M is enough

	while (!glfwWindowShouldClose(window))
	{
		// input
		processInput(window);

		// KERNEL HERE

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
	shader.setMat4("model", glm::mat4(1.0f));
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	// render objects
	for each (Object o in hostAquarium.objects)
	{
		float scaleX = (1.0f / AQUARIUM_WIDTH) * o.size;
		float scaleY = (1.0f / AQUARIUM_HEIGHT) * o.size;
		glm::mat4 scale = glm::mat4(1.0f);
		glm::mat4 trans = glm::mat4(1.0f);
		// SMTH IS WRONG WITH THIS - IDK 
		//trans = glm::translate(trans, glm::vec3(o.position.x - SCR_WIDTH/2, o.position.y - SCR_HEIGHT/2, 0));
		//scale = glm::scale(scale, glm::vec3(scaleX, scaleY, 1));
		shader.setMat4("mvp", trans*scale);

		if (o.fish)
		{
			// render fish
			glBindVertexArray(VAO_fish);
			glDrawElements(GL_TRIANGLES, 9, GL_UNSIGNED_INT, 0);
		}
		else
		{
			// render alga
			glBindVertexArray(VAO_alga);
			glDrawElements(GL_TRIANGLES, 18, GL_UNSIGNED_INT, 0);
		}
	}
}

// --------------------------------------------- MEMORY MANAGEMENT FUNCTIONS ----------------------------------------------
void initMemory()
{
	mallocHostAquariumStruct(MAXOBJCOUNT);
	mallocDeviceAquariumStruct(MAXOBJCOUNT);
	mallocDeviceSceneStruct();
}
void copyAquariumStructToDevice()
{
	int n = deviceAquariumStruct.objectsCount = hostAquariumStruct.objectsCount;

	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.positions.x, hostAquariumStruct.objects.positions.x, sizeof(float)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.positions.y, hostAquariumStruct.objects.positions.y, sizeof(float)*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.directionVecs.x, hostAquariumStruct.objects.directionVecs.x, sizeof(float)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.directionVecs.y, hostAquariumStruct.objects.directionVecs.y, sizeof(float)*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.alives, hostAquariumStruct.objects.alives, sizeof(bool)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.fish, hostAquariumStruct.objects.fish, sizeof(bool)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(deviceAquariumStruct.objects.sizes, hostAquariumStruct.objects.sizes, sizeof(float)*n, cudaMemcpyHostToDevice));
}
void copyAquariumStructFromDevice()
{
	int n = hostAquariumStruct.objectsCount = hostAquariumStruct.objectsCount;

	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.positions.x, deviceAquariumStruct.objects.positions.x, sizeof(float)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.positions.y, deviceAquariumStruct.objects.positions.y, sizeof(float)*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.directionVecs.x, deviceAquariumStruct.objects.directionVecs.x, sizeof(float)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.directionVecs.y, deviceAquariumStruct.objects.directionVecs.y, sizeof(float)*n, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.alives, deviceAquariumStruct.objects.alives, sizeof(bool)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.fish, deviceAquariumStruct.objects.fish, sizeof(bool)*n, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(hostAquariumStruct.objects.sizes, deviceAquariumStruct.objects.sizes, sizeof(float)*n, cudaMemcpyHostToDevice));
}
void mallocHostAquariumStruct(int objectsCount)
{
	hostAquariumStruct.objectsCount = objectsCount;
	hostAquariumStruct.objects.positions.x = (float*)malloc(objectsCount * sizeof(float));
	hostAquariumStruct.objects.positions.y = (float*)malloc(objectsCount * sizeof(float));
	hostAquariumStruct.objects.directionVecs.x = (float*)malloc(objectsCount * sizeof(float));
	hostAquariumStruct.objects.directionVecs.y = (float*)malloc(objectsCount * sizeof(float));
	hostAquariumStruct.objects.alives = (bool*)malloc(objectsCount * sizeof(bool));
	hostAquariumStruct.objects.fish = (bool*)malloc(objectsCount * sizeof(bool));
	hostAquariumStruct.objects.sizes = (float*)malloc(objectsCount * sizeof(float));
}
void mallocDeviceAquariumStruct(int objectsCount)
{
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.positions.x, sizeof(float)*objectsCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.positions.y, sizeof(float)*objectsCount));

	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.directionVecs.x, sizeof(float)*objectsCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.directionVecs.y, sizeof(float)*objectsCount));

	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.alives, sizeof(bool)*objectsCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.fish, sizeof(bool)*objectsCount));
	checkCudaErrors(cudaMalloc((void**)&deviceAquariumStruct.objects.sizes, sizeof(float)*objectsCount));
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
	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.positions.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.positions.y));

	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.directionVecs.x));
	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.directionVecs.y));

	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.alives));
	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.fish));
	checkCudaErrors(cudaFree(deviceAquariumStruct.objects.sizes));
}
void freeHostAquariumStruct()
{
	free(hostAquariumStruct.objects.positions.x);
	free(hostAquariumStruct.objects.positions.y);

	free(hostAquariumStruct.objects.directionVecs.x);
	free(hostAquariumStruct.objects.directionVecs.y);

	free(hostAquariumStruct.objects.alives);
	free(hostAquariumStruct.objects.fish);
	free(hostAquariumStruct.objects.sizes);
}
void freeDeviceSceneStruct()
{
	checkCudaErrors(cudaFree(deviceSceneStruct.objCellArray));
	checkCudaErrors(cudaFree(deviceSceneStruct.cellSizeArray));
}