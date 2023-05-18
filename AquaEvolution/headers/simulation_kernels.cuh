#ifndef SIMULATION_KERNELS_H
#define SIMULATION_KERNELS_H

#include "device_launch_parameters.h"
#include "structs/aquarium.cuh"
#include "scene.cuh"

/// <summary>
/// Kernel to perform decision phase calculation for fishes. In this kernel it is decided
/// upon wherther fish move in some direction or eat some algae
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void decision_fish(AquariumSoA aquarium, SceneSoA scene);

/// <summary>
/// Kernel to perform decision phase calculation for algae. 
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void decision_algae(AquariumSoA aquarium, SceneSoA scene);

/// <summary>
/// Kernel to perfom move phase calculation for fishes.
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void move_fish(AquariumSoA aquarium, SceneSoA scene);

/// <summary>
/// Kernel to perfom move phase calculation for algae.
/// </summary>
/// <param name="aquarium">Struct with objects in aquarium</param>
/// <param name="scene">some scene bullshit</param>
/// <returns>Nada</returns>
__global__ void move_algae(AquariumSoA aquarium, SceneSoA scene);

///// <summary>
///// Kernel to perform backet/radix/whatever sort on fishes and verify their position in aquarium
///// </summary>
///// <param name="aquarium">Struct with objects in aquarium</param>
///// <param name="scene">some scene bullshit</param>
///// <returns>Nada</returns>
//__global__ void sort_fish(AquariumSoA aquarium, SceneSoA scene);
//
///// <summary>
///// Kernel to perform backet/radix/whatever sort on algae and verify their position in aquarium
///// </summary>
///// <param name="aquarium">Struct with objects in aquarium</param>
///// <param name="scene">some scene bullshit</param>
///// <returns>Nada</returns>
//__global__ void sort_algae(AquariumSoA aquarium, SceneSoA scene);

#endif // !SIMULATION_KERNELS_H