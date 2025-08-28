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
		// For defining area of tile by two corners (top-left, bottom-right)
		struct Rectangle {
			uint64_t x1, y1, x2, y2;
		};
		template<typename T>
		struct Tile {
			Rectangle area;
			T* terrainPtr;
			std::vector<char> encodedData;
			void encode() {
				
			}
		};
		// Contains commands to be sent from main CPU thread to worker threads.
		template<typename T>
		struct TileCommand {
			enum CMD {
				CMD_NOOP,
				CMD_COPY_FROM_RAW_SOURCE_DATA,
				CMD_ENCODE_HUFFMAN,
			};
			CMD command;
			Rectangle tileSource;
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
			TileWorker(int index, T* terrainPtrPrm = nullptr) : commandQueue(), working(true), worker([&, index, terrainPtrPrm]() {
				bool workingTmp = true;
				{
					std::unique_lock<std::mutex> lock(mutex);
					exiting = false;
				}
				if (terrainPtrPrm != nullptr) {

					while (workingTmp) {
						{
							std::unique_lock<std::mutex> lock(mutex);
							cond.wait(lock);
						}
						std::queue<TileCommand<T>> localQueue;
						{
							std::unique_lock<std::mutex> lock(mutex);
							workingTmp = working;	
							localQueue.swap(commandQueue);
						}
						while (localQueue.size() > 0) {
							auto task = localQueue.front();
							localQueue.pop();
							if (task.command == TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN) {
								std::cout << "task..." << std::endl;
							}
						}
					}
					std::cout << commandQueue.size() << std::endl;
					std::unique_lock<std::mutex> lock(mutex);
					exiting = true;
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
		std::shared_ptr<Helper::Tile<T>> tiles;
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		TileManager(T* terrainPtr, uint64_t width, uint64_t height, uint64_t tileWidth, uint64_t tileHeight,  int numThreads = std::thread::hardware_concurrency(), int deviceId = 0) {
			deviceIndex = deviceId;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(i, terrainPtr);
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
				for (uint64_t x = 0; x < numTilesY; x++) {
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
		}

		~TileManager() {

		}
	};
}
#endif