#include <stdio.h>
#include <random>
#undef NDEBUG
#include <assert.h>
#include "CompressedTerrainCache.cuh"

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    
    size_t terrainWidth = 10000;
    size_t terrainHeight = 10000;
    size_t tileWidth = 100;
    size_t tileHeight = 100;
    size_t numTerrainElements = terrainWidth * terrainHeight;
    using T = unsigned char;
    // Generating sample terrain.
    std::shared_ptr<T> terrain = std::shared_ptr<T>(new T[numTerrainElements], [](T* ptr) { delete[] ptr; });
    unsigned char terrainTypes[8] = {
        '.', // sand
        '~', // water
        'o', // rock
        '$', // bridge
        ' ', // void
        'X', // iron gate
        '=', // magic chest
        '#', // leather armor
    };
    for (size_t y = 0; y < terrainHeight; y++) {
        for (size_t x = 0; x < terrainWidth; x++) {
            size_t index = x + y * terrainWidth;
            terrain.get()[index] = terrainTypes[rand() % 8];
        }
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0;
    int numCpuThreads = 20;
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, numCpuThreads, deviceIndex);
    // Testing if decoding works.
    tileManager.unitTestForDataIntegrity();
    // Benchmarking normal direct access for all tiles.
    tileManager.benchmarkNormalAccess();
    // Benchmarking decoded access for all tiles.
    tileManager.benchmarkDecodedAccess();
    return 0;
}