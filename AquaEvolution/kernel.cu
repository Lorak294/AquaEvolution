#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "headers/Shader.h"
#include "headers/rendering.cuh"
#include "headers/aquarium.cuh"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <cstdlib>
#include <ctime>
#include <random>

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
void createTexture();
void cleanup();
void renderLoop(GLFWwindow* window, Shader shader);
void copy_bitmap_to_host();
void makeNewGeneration();
void resetAquariumStruct(int algaeCount);

// settings
unsigned int SCR_WIDTH = 1600;
unsigned int SCR_HEIGHT = 900;
size_t BITMAP_RGB_SIZE = SCR_WIDTH * SCR_HEIGHT * sizeof(GLubyte) * 3;
const float alga::initaialSize = 0.5f;

const unsigned int TX = 16;
const unsigned int TY = 16;

const bool GPU_RNEDER = true;

GLubyte* hostBitmap;
GLubyte* deviceBitmap;

aquarium hostAquarium;
s_aquarium hostAquariumStruct;
s_aquarium deviceAquariumStruct;
int algaeCount = 1;

double oneFrameTime;
double copyTime;
double renderTime;

unsigned int VBO, VAO, EBO, tex;

int main()
{
	srand(static_cast<unsigned> (time(0)));

	GLFWwindow* window = createGLWindow();

	// create simple shader to display texture
	Shader shader("texture.vs", "texture.fs");

	// alocating bitmaps resources
	hostBitmap = (GLubyte*)malloc(BITMAP_RGB_SIZE);
	checkCudaErrors(cudaMalloc((void **)&deviceBitmap, BITMAP_RGB_SIZE));

	// creating openGL buffers and drawing
	createBuffers();
	createTexture();
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
		return;
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
	BITMAP_RGB_SIZE = SCR_WIDTH * SCR_HEIGHT * sizeof(GLubyte) * 3;

	//new texture
	//glTexImage2D()

	// resize btimaps
	free(hostBitmap);
	hostBitmap = (GLubyte*)malloc(BITMAP_RGB_SIZE);
	memset(hostBitmap, 0, BITMAP_RGB_SIZE);

	checkCudaErrors(cudaFree(deviceBitmap));
	checkCudaErrors(cudaMalloc((void **)&deviceBitmap, BITMAP_RGB_SIZE));
	cudaMemset(deviceBitmap, 0, BITMAP_RGB_SIZE);
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
	GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
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
	// set up vertex data (and buffer(s)) and configure vertex attributes
	float vertices[] = {
		// positions          // texture coords
		 1.0f,  1.0f, 0.0f,   1.0f, 1.0f, // top right
		 1.0f, -1.0f, 0.0f,   1.0f, 0.0f, // bottom right
		-1.0f, -1.0f, 0.0f,   0.0f, 0.0f, // bottom left
		-1.0f,  1.0f, 0.0f,   0.0f, 1.0f  // top left 
	};
	unsigned int indices[] = {
		0, 1, 3, // first triangle
		1, 2, 3  // second triangle
	};


	glGenVertexArrays(1, &VAO);
	glGenBuffers(1, &VBO);
	glGenBuffers(1, &EBO);

	glBindVertexArray(VAO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// texture coord attribute
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
}
void createTexture()
{
	// load and create a texture 
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex); // all upcoming GL_TEXTURE_2D operations now have effect on this texture object
	// set the texture wrapping parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);	// set texture wrapping to GL_REPEAT (default wrapping method)
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// set texture filtering parameters
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
}

// --------------------------------------------- MEMORY MANAGEMENT FUNCTIONS ------------------------------------
void cleanup()
{
	// de-allocate all resources once they've outlived their purpose:
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(1, &VBO);
	glDeleteBuffers(1, &EBO);
	glDeleteTextures(1, &tex);

	// glfw: terminate, clearing all previously allocated GLFW resources.
	glfwTerminate();

	// de-allocate bitmap buffers
	free(hostBitmap);
	checkCudaErrors(cudaFree(deviceBitmap));
}
void copy_bitmap_to_host()
{
	checkCudaErrors(cudaMemcpy(hostBitmap, deviceBitmap, BITMAP_RGB_SIZE, cudaMemcpyDeviceToHost));
}

// --------------------------------------------- RENDERING FUNCTIONS ----------------------------------------------
void renderLoop(GLFWwindow* window, Shader shader)
{
	while (!glfwWindowShouldClose(window))
	{
		// start timestamp
		oneFrameTime = glfwGetTime();

		// allocate aquarium resources


		// input
		processInput(window);

		// render
		glClear(GL_COLOR_BUFFER_BIT);

		if (GPU_RNEDER)
		{
			dim3 blocks(SCR_WIDTH / TX + 1, SCR_HEIGHT / TY + 1);
			dim3 threads(TX, TY);

			// render kernel function
			renderTime = glfwGetTime();
			render_GPU << <blocks, threads >> > (deviceBitmap, SCR_WIDTH, SCR_HEIGHT);
			checkCudaErrors(cudaGetLastError());
			checkCudaErrors(cudaDeviceSynchronize());
			renderTime = glfwGetTime() - renderTime;

			// copy bitmap to host
			copyTime = glfwGetTime();
			copy_bitmap_to_host();
			copyTime = glfwGetTime() - copyTime;
		}
		else
		{
			render_CPU(hostBitmap, SCR_WIDTH, SCR_HEIGHT);
		}

		// create texture form bitmap
		glBindTexture(GL_TEXTURE_2D, tex);
		//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, SCR_WIDTH, SCR_HEIGHT, GL_RGB, GL_UNSIGNED_BYTE, hostBitmap);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCR_WIDTH, SCR_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, hostBitmap);

		// render container
		shader.use();
		glBindVertexArray(VAO);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		// glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
		glfwSwapBuffers(window);
		glfwPollEvents();

		// end timestamp
		oneFrameTime = glfwGetTime() - oneFrameTime;
		glfwSetWindowTitle(window,
			("One Frame time: " + std::to_string(oneFrameTime) +
				"s; Copy time: " + std::to_string(copyTime) +
				"s; Render time: " + std::to_string(renderTime) +
				"s;"
				).c_str());

	}
}

// --------------------------------------------- AQUARIUM MANAGEMENT FUNCTIONS ----------------------------------------------

void makeNewGeneration()
{
	// copy alive algae to new aquarium
	hostAquarium.readFromDeviceStruct(hostAquariumStruct, false);

	// make new algae from each one
	hostAquarium.newGeneration();

	// apply mutations and store algae in new arrays
	resetAquariumStruct(hostAquarium.algae.size());
	hostAquarium.writeToDeviceStruct(hostAquariumStruct);
}


void resetAquariumStruct(int algaeCount)
{
	hostAquariumStruct.algaeCount = algaeCount;
	
	free(hostAquariumStruct.algae.positions.x);
	free(hostAquariumStruct.algae.positions.y);
	free(hostAquariumStruct.algae.driftingVecs.x);
	free(hostAquariumStruct.algae.driftingVecs.y);
	free(hostAquariumStruct.algae.alives);
	free(hostAquariumStruct.algae.sizes);

	hostAquariumStruct.algae.positions.x = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.positions.y = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.driftingVecs.x = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.driftingVecs.y = (float*)malloc(algaeCount * sizeof(float));
	hostAquariumStruct.algae.alives = (bool*)malloc(algaeCount * sizeof(bool));
	hostAquariumStruct.algae.sizes = (float*)malloc(algaeCount * sizeof(float));
}