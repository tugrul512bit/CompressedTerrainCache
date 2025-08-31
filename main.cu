#include <stdio.h>
#include <random>
#include "CompressedTerrainCache.cuh"

int main()
{
    
    size_t terrainWidth = 20003;
    size_t terrainHeight = 14005;
    size_t tileWidth = 210;
    size_t tileHeight = 200;
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
        '#' // leather armor
    };
    for (size_t y = 0; y < terrainHeight; y++) {
        for (size_t x = 0; x < terrainWidth; x++) {
            size_t index = x + y * terrainWidth;
            terrain.get()[index] = terrainTypes[rand() % 8];
        }
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0; // 0 means first cuda gpu, 1 means second cuda gpu, ...
    int numCpuThreads = 20; // can have up to concurrency limit number of cpu threads.
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, numCpuThreads, deviceIndex);
    // Testing if decoding works.
    tileManager.unitTestForDataIntegrity();
    tileManager.unitTestForDataIntegrity();
    tileManager.unitTestForDataIntegrity();
    tileManager.unitTestForDataIntegrity();
    // Benchmarking normal direct access for all tiles.
    tileManager.benchmarkNormalAccess();
    // Benchmarking decoded access for all tiles.
    tileManager.benchmarkDecodedAccess();
    // Testing if selected tile decoding works.
    // Selected tiles: 10, 25, 44, 45 where consecutive values are on same row of tiles except at edges.
    tileManager.unitTestForSelectedDataIntegrity({ 10, 25, 44, 45 });
    // Benchmarking for custom selection of tiles (using player position + visibility range vs terrain coordinates)
    uint32_t playerX = terrainWidth / 2; // player is on middle of terrain.
    uint32_t playerY = terrainHeight / 2;
    uint32_t playerVisibilityRadius = 2500; // player can see 2500 units far.
    uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth; // internally this calculation is used as ordering of tiles.(index = tileX + tileY * numTilesX)
    uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
    uint32_t numTiles = numTilesX * numTilesY;
    std::vector<uint32_t> tileIndexList;
    for (uint32_t tileY = 0; tileY < numTilesY; tileY++) {
        for (uint32_t tileX = 0; tileX < numTilesX; tileX++) {
            // Checking if player visibility range collides with the current tile.
            uint32_t distanceX = playerX - (tileX * tileWidth + tileWidth / 2);
            uint32_t distanceY = playerY - (tileY * tileHeight + tileHeight / 2);
            uint32_t distance = sqrt(distanceX * distanceX + distanceY * distanceY);
            if (distance < playerVisibilityRadius) {
                tileIndexList.push_back(tileX + tileY * numTilesX);
            }
        }
    }
    tileManager.benchmarkForSelectedTilesNormalAccess(tileIndexList);
    tileManager.benchmarkForSelectedTilesEncodedAccess(tileIndexList);
    return 0;
}