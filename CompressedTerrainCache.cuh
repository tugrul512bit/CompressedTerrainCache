#ifndef _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#define _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#include <vector>
#include <thread>
#include <memory>
#include <queue>
#include <mutex>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include "HuffmanTileEncoder.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(err);                                                     \
        }                                                                  \
    } while (0)
namespace CompressedTerrainCache {
	namespace Helper {
		// Contains commands to be sent from main CPU thread to worker threads.
		template<typename T>
		struct TileCommand {
			enum CMD {
				CMD_NOOP,
				CMD_COPY_FROM_RAW_SOURCE_DATA,
				CMD_ENCODE_HUFFMAN,
			};
			CMD command;
			HuffmanTileEncoder::Rectangle tileSource;
		};
		template<typename T>
		struct TileWorker {
			std::queue<TileCommand<T>> commandQueue;
			bool working;
			bool exiting;
			std::mutex mutex;
			std::condition_variable cond;
			std::thread worker;
			T* terrainPtr;
			int id;
			TileWorker(int index, T* terrainPtrPrm = nullptr, std::shared_ptr<std::vector<HuffmanTileEncoder::Tile<T>>> tilesPtr = nullptr, std::shared_ptr<std::mutex> tilesLockPtr = nullptr) : commandQueue(), working(true), worker([&, index, terrainPtrPrm, tilesPtr, tilesLockPtr]() {
				bool workingTmp = true;
				std::queue<TileCommand<T>> localCommandQueue;
				std::vector<HuffmanTileEncoder::Tile<T>> localTiles;
				{
					std::unique_lock<std::mutex> lock(mutex);
					exiting = false;
				}
				if (terrainPtrPrm != nullptr && tilesPtr != nullptr && tilesLockPtr != nullptr) {

					while (workingTmp) {
						{
							std::unique_lock<std::mutex> lock(mutex);
							cond.wait(lock);
						}

						{
							std::unique_lock<std::mutex> lock(mutex);
							workingTmp = working;	
							localCommandQueue.swap(commandQueue);
						}
						while (localCommandQueue.size() > 0) {
							auto task = localCommandQueue.front();
							localCommandQueue.pop();
							if (task.command == TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN) {
								HuffmanTileEncoder::Tile<T> currentTile;
								localTiles.push_back(currentTile);
							}
						}
						if(localTiles.size() > 0)
						{
							{
								std::unique_lock<std::mutex> lock(*tilesLockPtr);
								if (tilesPtr != nullptr) {
									tilesPtr->insert(tilesPtr->end(), localTiles.begin(), localTiles.end());
								}
							}
							localTiles.clear();
						}
					}
					std::unique_lock<std::mutex> lock(mutex);
					exiting = true;
					std::cout << "r=" << localCommandQueue.size() << std::endl;
				}
			}), terrainPtr(terrainPtrPrm), id(index) { }

			void addCommand(TileCommand<T> cmd) {
				{
					std::unique_lock<std::mutex> lock(mutex);
					commandQueue.push(cmd);
				}
				cond.notify_one();
			}

			~TileWorker() {
				if (terrainPtr != nullptr) {

					bool exitingTmp = false;
					while(!exitingTmp)
					{
						{
							std::unique_lock<std::mutex> lock(mutex);
							exitingTmp = exiting;
						}
						if(!exitingTmp)
						{
							std::unique_lock<std::mutex> lock(mutex);
							working = false;
						}
						cond.notify_one();
						std::this_thread::yield();
					}
					worker.join();
				}
			}
		};

		struct UnifiedMemory {
			std::shared_ptr<char> ptr;
			UnifiedMemory(uint64_t sizeBytes = 0) {
				if (sizeBytes == 0) {
					ptr = nullptr;
				}	else {
					char* tmp;
					CUDA_CHECK(cudaMallocManaged(&tmp, sizeBytes, cudaMemAttachGlobal));
					ptr = std::shared_ptr<char>(tmp, [](char* ptr) { CUDA_CHECK(cudaFree(ptr)); });
				}
			}
			~UnifiedMemory() {

			}
		};
	}
	/* 
	Starts dedicated threads for continuous encoding of many tiles with asynchronous reading from gpu.
	Uses CUDA unified memory as staging buffer for uploading randomly positioned fine-grained tiles from host to device from much larger storage (RAM).
	*/
	template<typename T>
	struct TileManager {
		int deviceIndex;
		std::shared_ptr<Helper::UnifiedMemory> memory;
		std::shared_ptr<std::mutex> tilesLock;
		std::shared_ptr<std::vector<HuffmanTileEncoder::Tile<T>>> tiles;
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		TileManager(T* terrainPtr, uint64_t width, uint64_t height, uint64_t tileWidth, uint64_t tileHeight,  int numThreads = std::thread::hardware_concurrency(), int deviceId = 0) {
			deviceIndex = deviceId;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			tiles = std::make_shared<std::vector<HuffmanTileEncoder::Tile<T>>>();
			tilesLock = std::make_shared<std::mutex>();
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(i, terrainPtr, tiles, tilesLock);
				workers.push_back(worker);
			}
			uint64_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint64_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint64_t numTiles = numTilesX * numTilesY;
			memory = std::make_shared<Helper::UnifiedMemory>(numTiles * tileWidth * tileHeight * sizeof(T));
			std::cout << "Encoding tiles..." << std::endl;
			int idx = 0;
			int numWorkers = workers.size();
			for (uint64_t y = 0; y < numTilesY; y++) {
				for (uint64_t x = 0; x < numTilesX; x++) {
					int index = idx++ % numWorkers;
					Helper::TileCommand<T> cmd;
					cmd.command = Helper::TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN;
					cmd.tileSource.x1 = x * tileWidth;
					cmd.tileSource.y1 = y * tileHeight;
					cmd.tileSource.x2 = x * tileWidth + tileWidth;
					cmd.tileSource.y2 = y * tileHeight + tileHeight;
					workers[index]->addCommand(cmd);
				}
			}
			std::cout << "m=" << idx << std::endl;
		}

		~TileManager() {
			workers.clear();
			std::unique_lock<std::mutex> lock(*tilesLock);
			std::cout << "n=" << tiles->size() << std::endl;
		}
	};
}
#endif