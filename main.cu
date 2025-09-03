#include <stdio.h>
#include <random>
#include <cmath>
// Cached encoder tool for 2d terrain.
#include "CompressedTerrainCache.cuh"
// OpenCV4 for 2d render.
#include<opencv2/opencv.hpp>

int main()
{
    // Player can see this far.
    uint64_t playerVisibilityRadius = 1600;
    // 2D terrain map size, in units.
    uint64_t terrainWidth = 16384;
    uint64_t terrainHeight = 16384;
    // 2D tile size, in units.
    uint64_t tileWidth = 256;
    uint64_t tileHeight = 256;
    // Tile cache size, in tiles (so that 32x32 cache can store 1024 tiles at once)
    uint64_t tileCacheSlotColumns = 32;
    uint64_t tileCacheSlotRows = 32;
    // internally this calculation is used as ordering of tiles.(index = tileX + tileY * numTilesX) (row-major)
    uint64_t numTerrainElements = terrainWidth * terrainHeight;
    uint64_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
    uint64_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
    uint64_t numTiles = numTilesX * numTilesY;

    // Terrain element type (Has to be POD type)
    //using T = unsigned char;
    using T = uint32_t;

    // Generating sample terrain (2D cos wave pattern).
    std::shared_ptr<T> terrain = std::shared_ptr<T>(new T[numTerrainElements], [](T* ptr) { delete[] ptr; });
    for (uint64_t y = 0; y < terrainHeight; y++) {
        for (uint64_t x = 0; x < terrainWidth; x++) {
            uint64_t index = x + y * terrainWidth;
            unsigned char blue = 77 + cos(x * 0.002f) * cos(y * 0.002f) * 50;
            unsigned char green = 37 + cos(x * 0.0005f) * cos(y * 0.0005f) * 20;
            unsigned char red = 130 + cos(x * 0.0004f) * cos(y * 0.0004f) * 100;
            unsigned char alpha = 255;
            terrain.get()[index] = ((sizeof(T) == 4) ? (blue | (green << 8) | (red << 16) | (alpha << 24)) : blue);
        }
    }

    // Creating tile manager that uses terrain as input.
    int deviceIndex = 0; // 0 means first cuda gpu, 1 means second cuda gpu, ...
    int numCpuThreads = 20; // can have up to concurrency limit number of cpu threads.
    CompressedTerrainCache::TileManager<T> tileManager(terrain.get(), terrainWidth, terrainHeight, tileWidth, tileHeight, tileCacheSlotColumns, tileCacheSlotRows,  numCpuThreads, deviceIndex);

    // Rendering reference terrain in a window.
    cv::namedWindow("Downscaled Raw Terrain Data");
    cv::resizeWindow("Downscaled Raw Terrain Data", 1024, 1024);
    cv::Mat img(terrainHeight, terrainWidth, sizeof(T) == 4 ? CV_8UC4 : CV_8UC1, terrain.get());
    cv::Mat downScaledImg;
    cv::resize(img, downScaledImg, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
    cv::imshow("Downscaled Raw Terrain Data", downScaledImg);
    cv::waitKey(1);
    cv::namedWindow("Loaded Tiles");
    cv::resizeWindow("Loaded Tiles", 1024, 1024);



    float angle = 0.0f;
    double timeNormalAccess = 0.0f;
    double timeDecode = 0.0f;
    double dataSizeNormalAccess = 0.0f;
    double dataSizeDecode = 0.0f;
    double throughputNormalAccess = 0.0f;
    double throughputDecode = 0.0;
    unsigned char* loadedTilesOnDevice_d = nullptr;
    constexpr int ACCESS_METHOD_DIRECT = 0;
    constexpr int ACCESS_METHOD_DECODE_HUFFMAN = 1;
    int accessMethod = 0;
    // Sample game loop.
    while (true) {
        angle += 0.01f;
        // Creating a sample list of tile-indices using visibility range of player.
        std::vector<uint32_t> tileIndexList;
        for (uint64_t tileY = 0; tileY < numTilesY; tileY++) {
            for (uint64_t tileX = 0; tileX < numTilesX; tileX++) {
                // Checking if player visibility range collides with the current tile.
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
            case ACCESS_METHOD_DIRECT: loadedTilesOnDevice_d = tileManager.accessSelectedTiles(tileIndexList, &timeNormalAccess, &dataSizeNormalAccess, &throughputNormalAccess); break;
            case ACCESS_METHOD_DECODE_HUFFMAN:loadedTilesOnDevice_d = tileManager.decodeSelectedTiles(tileIndexList, &timeDecode, &dataSizeDecode, &throughputDecode); break;
            default:break;
        }

        uint64_t outputBytes = tileIndexList.size() * (size_t)tileWidth * tileHeight * sizeof(T);
        std::vector<T> loadedTilesOnHost_h(tileIndexList.size() * (size_t)tileWidth * tileHeight);
        // Downloading output tile data from device memory to RAM.
        CUDA_CHECK(cudaMemcpy(loadedTilesOnHost_h.data(), loadedTilesOnDevice_d, outputBytes, cudaMemcpyDeviceToHost));
        // Clearing old terrain to see if visibility range works correctly.
        std::fill(terrain.get(), terrain.get() + (terrainWidth * terrainHeight), sizeof(T) == 1 ? 255 : 0);
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
        cv::Mat img2(terrainHeight, terrainWidth, sizeof(T) == 4 ? CV_8UC4 : CV_8UC1, terrain.get());
        cv::Mat downScaledImg2;
        cv::resize(img2, downScaledImg2, cv::Size(1024, 1024), 0, 0, cv::INTER_AREA);
        std::string directMethod = std::string("Unified memory tile stream:");
        std::string decodeInfo1 = std::string("Kernel = ") + std::to_string(timeNormalAccess) + std::string(" seconds");
        std::string decodeInfo2 = std::string("Data = ") + std::to_string(dataSizeNormalAccess) + std::string(" GB");
        std::string decodeInfo3 = std::string("Throughput = ") + std::to_string(throughputNormalAccess) + std::string(" GB/s");
        std::string decodeMethod = std::string("Unified memory encoded-tile stream + decoding + 2D caching:");
        std::string decodeInfo4 = std::string("Kernel = ") + std::to_string(timeDecode) + std::string(" seconds");
        std::string decodeInfo5 = std::string("Data = ") + std::to_string(dataSizeDecode) + std::string(" GB");
        std::string decodeInfo6 = std::string("Throughput = ") + std::to_string(throughputDecode) + std::string(" GB/s");
        cv::Mat benchmark;

        cv::putText(downScaledImg2, directMethod, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo1, cv::Point(20, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo2, cv::Point(20, 100), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo3, cv::Point(20, 120), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeMethod, cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo4, cv::Point(20, 180), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo5, cv::Point(20, 200), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, decodeInfo6, cv::Point(20, 220), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::putText(downScaledImg2, "Press ESC to exit", cv::Point(20, 980), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        cv::imshow("Loaded Tiles", downScaledImg2);
        int key = cv::waitKey(1);
        if (key == 27) {
            break;
        }
    }
    cv::destroyAllWindows();
    return 0;
}