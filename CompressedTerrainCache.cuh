#ifndef _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#define _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#include <vector>
#include <thread>
#include <memory>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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
		// Generates tiles from raw terrain data.
		struct TileGenerator {

		};

		// Encodes tile with Huffman Encoding.
		struct TileCompressor {

		};
		// For defining position and size of a tile.
		struct Rectangle {
			uint64_t x1, x2, y1, y2;
		};
		// Contains commands to be sent from main CPU thread to worker threads.
		template<typename T>
		struct TileCommand {
			enum CMD {
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
			std::mutex mutex;
			std::condition_variable cond;
			std::thread worker;
			T* terrainPtr;

			TileWorker(T* terrainPtrPrm = nullptr) : commandQueue(), working(true), worker([&]() {
				bool workingTmp = true;
				std::cout << "thread "<<(uint64_t)terrainPtrPrm << std::endl;
				while (workingTmp) {
					{
						std::unique_lock<std::mutex> lock(mutex);
						cond.wait(lock);
					}
					std::lock_guard<std::mutex> lg(mutex);
					workingTmp = working;
				}
				std::cout << " x " << std::endl;
			}), terrainPtr(terrainPtrPrm) { }

			void addCommand(TileCommand<T> cmd) {
				std::lock_guard<std::mutex> lg(mutex);
				commandQueue.push(cmd);
				cond.notify_one();
			}

			~TileWorker() {
				{
					std::lock_guard<std::mutex> lg(mutex);
					working = false;
					cond.notify_one();
				}
				worker.join();
			}
		};

		struct UnifiedMemory {
			char* ptr;
			UnifiedMemory(uint64_t sizeBytes = 0) {
				if (sizeBytes == 0) {
					ptr = nullptr;
				}	else {
					CUDA_CHECK(cudaMallocManaged(&ptr, sizeBytes, cudaMemAttachGlobal));
				}
			}
			~UnifiedMemory() {
				if (ptr != nullptr) {
					CUDA_CHECK(cudaFree(ptr));
				}
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
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		std::shared_ptr<Helper::UnifiedMemory> memory;
		TileManager(T* terrainPtr, int numThreads = std::thread::hardware_concurrency(), int deviceId = 0) {
			deviceIndex = deviceId;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(terrainPtr);
				workers.push_back(worker);
			}
		}

		~TileManager() {

		}
	};
}
#endif