#include <stdio.h>
#include <random>
#include <cmath>
// Cached encoder tool for 2d terrain.
#include "CompressedTerrainCache.cuh"
// OpenCV4 for 2d render.
#include<opencv2/opencv.hpp>

int main()
{
    uint64_t playerVisibilityRadius = 1500; // player can see this far.
    uint64_t terrainWidth = 15 * 1024;
    uint64_t terrainHeight = 15 * 1024;
    uint64_t tileWidth = 256;
    uint64_t tileHeight = 256;
    uint64_t numTerrainElements = terrainWidth * terrainHeight;
    uint64_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth; // internally this calculation is used as ordering of tiles.(index = tileX + tileY * numTilesX)
    uint64_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
    uint64_t numTiles = numTilesX * numTilesY;
    using T = unsigned char;
    // Generating sample terrain (2D cos wave pattern).
    std::shared_ptr<T> terrain = std::shared_ptr<T>(new T[numTerrainElements], [](T* ptr) { delete[] ptr; });
    for (uint64_t y = 0; y < terrainHeight; y++) {
        for (uint64_t x = 0; x < terrainWidth; x++) {
            uint64_t index = x + y * terrainWidth;
            unsigned char color = 77 + cos(x * 0.003f) * cos(y * 0.003f) * 50;
            terrain.get()[index] = color;
        }
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0; // 0 means first cuda gpu, 1 means second cuda gpu, ...
    int numCpuThreads = 20; // can have up to concurrency limit number of cpu threads.
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, numCpuThreads, deviceIndex);

    // Rendering reference terrain in a window.
    cv::namedWindow("Downscaled Raw Terrain Data");
    cv::resizeWindow("Downscaled Raw Terrain Data", 1024, 1024);
    cv::Mat img(terrainHeight, terrainWidth, CV_8UC1, tileManager.memoryForOriginalTerrain.ptr.get());
    cv::Mat downScaledImg;
    cv::resize(img, downScaledImg, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
    cv::imshow("Downscaled Raw Terrain Data", downScaledImg);
    cv::waitKey(1);
    cv::namedWindow("Loaded Tiles");
    cv::resizeWindow("Loaded Tiles", 1024, 1024);



    float angle = 0.0f;
    double timeNormalAccess = 0.0f;
    double timeDecode = 0.0f;
    unsigned char* loadedTilesOnDevice_d = nullptr;
    constexpr int ACCESS_METHOD_DIRECT = 0;
    constexpr int ACCESS_METHOD_DECODE_HUFFMAN = 1;
    int accessMethod = 0;
    // Sample game loop.
    while (true) {
        // Creating a sample list of tile-indices using visibility range of player.
        std::vector<uint32_t> tileIndexList;
        for (uint64_t tileY = 0; tileY < numTilesY; tileY++) {
            for (uint64_t tileX = 0; tileX < numTilesX; tileX++) {
                // Checking if player visibility range collides with the current tile.
                angle += 0.000005f;
                uint64_t playerX = terrainWidth / 2 + cos(angle) * terrainWidth / 4;
                uint64_t playerY = terrainHeight / 2 + sin(angle) * terrainHeight / 4;
                uint64_t distanceX = playerX - (tileX * tileWidth + tileWidth / 2);
                uint64_t distanceY = playerY - (tileY * tileHeight + tileHeight / 2);
                uint64_t distance = sqrt(distanceX * distanceX + distanceY * distanceY);
                if (distance < playerVisibilityRadius) {
                    tileIndexList.push_back(tileX + tileY * numTilesX);
                }
            }
        }

        accessMethod = 1 - accessMethod;
        switch (accessMethod) {
            case ACCESS_METHOD_DIRECT: loadedTilesOnDevice_d = tileManager.accessSelectedTiles(tileIndexList, &timeNormalAccess); break;
            case ACCESS_METHOD_DECODE_HUFFMAN:loadedTilesOnDevice_d = tileManager.decodeSelectedTiles(tileIndexList, &timeDecode); break;
            default:break;
        }

        uint64_t outputBytes = tileIndexList.size() * (size_t)tileWidth * tileHeight * sizeof(T);
        std::vector<unsigned char> loadedTilesOnHost_h(outputBytes);
        // Downloading output tile data.
        CUDA_CHECK(cudaMemcpy(loadedTilesOnHost_h.data(), loadedTilesOnDevice_d, outputBytes, cudaMemcpyDeviceToHost));
        // Clearing old terrain to see if visibility range works correctly.
        std::fill(terrain.get(), terrain.get() + (terrainWidth * terrainHeight), 0);
        uint32_t numErrors = 0;
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

        // Rendering benchmark window.
        cv::Mat img2(terrainHeight, terrainWidth, CV_8UC1, terrain.get());
        cv::Mat downScaledImg2;
        cv::resize(img2, downScaledImg2, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
        std::string directMethod = std::string("Unified memory tile stream = ") + std::to_string(timeNormalAccess) + std::string(" seconds");
        std::string decodeMethod = std::string("Unified memory encoded-tile stream + decoding = ") + std::to_string(timeDecode) + std::string(" seconds");
        cv::Mat benchmark;
        cv::cvtColor(downScaledImg2, benchmark, cv::COLOR_GRAY2BGR);
        cv::putText(benchmark, directMethod, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(benchmark, decodeMethod, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(benchmark, "Press ESC to exit", cv::Point(20, 980), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::imshow("Loaded Tiles", benchmark);
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}