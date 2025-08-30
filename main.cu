#include <stdio.h>
#include <random>
#include "CompressedTerrainCache.cuh"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

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
    unsigned char table[7] = { 'n', 'v', 'i', 'd', 'i', 'a', ' ' };
    for (size_t i = 0; i < numTerrainElements; i++) {
        terrain.get()[i] = table[i % 7];
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0;
    int numCpuThreads = 1;
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, numCpuThreads, deviceIndex);
    // Testing if decoding works.
    tileManager.unitTestForDataIntegrity();
    // Benchmarking normal direct access for all tiles.
    tileManager.benchmarkNormalAccess();
    // Benchmarking decoded access for all tiles.
    tileManager.benchmarkDecodedAccess();
    return 0;
}