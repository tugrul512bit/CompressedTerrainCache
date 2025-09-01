#include <stdio.h>
#include <random>
#include <cmath>
// Cached encoder tool for 2d terrain.
#include "CompressedTerrainCache.cuh"
// OpenCV4 for 2d render.
#include<opencv2/opencv.hpp>

int main()
{
    size_t terrainWidth = 1024 * 10;
    size_t terrainHeight = 1024 * 10;
    size_t tileWidth = 128;
    size_t tileHeight = 128;
    size_t numTerrainElements = terrainWidth * terrainHeight;
    using T = unsigned char;
    // Generating sample terrain (2D cos wave pattern).
    std::shared_ptr<T> terrain = std::shared_ptr<T>(new T[numTerrainElements], [](T* ptr) { delete[] ptr; });
    for (size_t y = 0; y < terrainHeight; y++) {
        for (size_t x = 0; x < terrainWidth; x++) {
            size_t index = x + y * terrainWidth;
            unsigned char color = 128 + cos(x * 0.003f) * cos(y * 0.003f) * 127;
            terrain.get()[index] = color;
        }
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0; // 0 means first cuda gpu, 1 means second cuda gpu, ...
    int numCpuThreads = 20; // can have up to concurrency limit number of cpu threads.
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, numCpuThreads, deviceIndex);

    // Testing if decoding works.
    tileManager.unitTestForDataIntegrity();
    cv::namedWindow("Downscaled Raw Terrain Data");
    cv::resizeWindow("Downscaled Raw Terrain Data", 1024, 1024);
    cv::Mat img(terrainHeight, terrainWidth, CV_8UC1, tileManager.memoryForOriginalTerrain.ptr.get());
    cv::Mat downScaledImg;
    cv::resize(img, downScaledImg, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
    cv::imshow("Downscaled Raw Terrain Data", downScaledImg);
    cv::waitKey(10000);
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
    // For testing >4GB data, adding the last tile.
    std::cout << "number of tiles = " << numTiles << std::endl;
    tileIndexList.push_back(numTiles - 1);
    tileManager.benchmarkForSelectedTilesNormalAccess(tileIndexList);
    unsigned char* loadedTilesOnDevice_d = tileManager.benchmarkForSelectedTilesEncodedAccess(tileIndexList);
    uint64_t outputBytes = tileIndexList.size() * (size_t)tileWidth * tileHeight * sizeof(T);
    std::vector<unsigned char> loadedTilesOnHost_h(outputBytes);
    // Downloading output tile data.
    CUDA_CHECK(cudaMemcpy(loadedTilesOnHost_h.data(), loadedTilesOnDevice_d, outputBytes, cudaMemcpyDeviceToHost));
    // Clearing old terrain to see if visibility range works correctly.
    std::fill(terrain.get(), terrain.get() + (terrainWidth * terrainHeight), 0);
    uint32_t tileIndexInOutput = 0;
    for (uint32_t tileIndex : tileIndexList) {
        uint32_t tileX = tileIndex % numTilesX;
        uint32_t tileY = tileIndex / numTilesX;
        for (uint32_t y = 0; y < tileHeight; y++) {
            for (uint32_t x = 0; x < tileWidth; x++) {
                uint64_t terrainX = (tileX * tileWidth + x);
                uint64_t terrainY = (tileY * tileHeight + y);
                uint64_t terrainDestinationIndex = terrainX + terrainY * (uint64_t)terrainWidth;
                uint64_t sourceIndex = tileIndexInOutput * (uint64_t)tileWidth * tileHeight + x + y * tileWidth;
                if (terrainX < terrainWidth && terrainY < terrainHeight) {
                    terrain.get()[terrainDestinationIndex] = loadedTilesOnHost_h[sourceIndex];
                }
            }
        }
        tileIndexInOutput++;
    }

    cv::namedWindow("Loaded Tiles");
    cv::resizeWindow("Loaded Tiles", 1024, 1024);
    cv::Mat img2(terrainHeight, terrainWidth, CV_8UC1, terrain.get());
    cv::Mat downScaledImg2;
    cv::resize(img2, downScaledImg2, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
    cv::imshow("Loaded Tiles", downScaledImg2);
    cv::waitKey(10000);
    cv::pollKey();
    cv::destroyAllWindows();
    return 0;
}