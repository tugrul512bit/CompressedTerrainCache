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
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.
		__global__ void k_decodeTile(unsigned char* encodedTiles, unsigned char* encodedTrees, int blockAlignedSize);
	}
	namespace Helper {
		struct UnifiedMemory {
			std::shared_ptr<unsigned char> ptr;
			UnifiedMemory(uint64_t sizeBytes = 0) {
				if (sizeBytes == 0) {
					ptr = nullptr;
				}
				else {
					unsigned char* tmp;
					CUDA_CHECK(cudaMallocManaged(&tmp, sizeBytes, cudaMemAttachGlobal));
					ptr = std::shared_ptr<unsigned char>(tmp, [](unsigned char* ptr) { CUDA_CHECK(cudaFree(ptr)); });
				}
			}
			~UnifiedMemory() {

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
			HuffmanTileEncoder::Rectangle tileSource;
			int index;
		};
		template<typename T>
		struct TileWorker {
			std::queue<TileCommand<T>> commandQueue;
			bool working;
			bool exiting;
			bool busy;
			std::mutex mutex;
			std::condition_variable cond;
			std::thread worker;
			T* terrainPtr;
			int id;
			std::vector<HuffmanTileEncoder::Tile<T>> localTiles;
			std::queue<TileCommand<T>> localCommandQueue;
			TileWorker(int deviceIndex, int index, T* terrainPtrPrm, uint64_t terrainWidth, uint64_t terrainHeight, 
				UnifiedMemory encodedTiles, 
				UnifiedMemory encodedTrees,
				std::shared_ptr<std::mutex> tilesLockPtr = nullptr) 
				: commandQueue(), working(true), worker([&, index, terrainPtrPrm, encodedTiles, encodedTrees, tilesLockPtr, terrainWidth, terrainHeight, deviceIndex]() {
				bool workingTmp = true;
				{
					std::unique_lock<std::mutex> lock(mutex);
					exiting = false;
					busy = true;
				}
				if (terrainPtrPrm != nullptr && tilesLockPtr != nullptr) {

					while (workingTmp) {
						{
							std::unique_lock<std::mutex> lock(mutex);
							if (!busy) {
								cond.wait(lock);
							}
						}
						{
							std::unique_lock<std::mutex> lock(mutex);
							workingTmp = working;
							localCommandQueue.swap(commandQueue);
						}
						while (localCommandQueue.size() > 0) {
							TileCommand<T> task = localCommandQueue.front();
							localCommandQueue.pop();
							if (task.command == TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN) {
								HuffmanTileEncoder::Tile<T> currentTile;
								currentTile.index = task.index;
								currentTile.area = task.tileSource;
								currentTile.copyInput(terrainWidth, terrainHeight, terrainPtrPrm);
								currentTile.encode();
								localTiles.push_back(currentTile);
							}
						}
						if(localTiles.size() > 0)
						{
							for (HuffmanTileEncoder::Tile<T>& tile : localTiles) {
								tile.copyOutput(encodedTiles.ptr.get(), encodedTrees.ptr.get());
							}
							{
								std::unique_lock<std::mutex> lock(mutex);
								if (commandQueue.empty()) {
									busy = false;
								}
							}
							localTiles.clear();
						}
					}
					std::unique_lock<std::mutex> lock(mutex);
					exiting = true;
				}
			}), terrainPtr(terrainPtrPrm), id(index) { }

			void addCommand(TileCommand<T> cmd) {
				{
					std::unique_lock<std::mutex> lock(mutex);
					commandQueue.push(cmd);
					busy = true;
				}
				cond.notify_one();
			}
			void wait() {
				bool busyTmp = true;
				while (busyTmp) {
					{
						std::unique_lock<std::mutex> lock(mutex);
						busyTmp = busy;
					}
					cond.notify_one();
					std::this_thread::yield();
				}
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
	}
	/* 
	Starts dedicated threads for continuous encoding of many tiles with asynchronous reading from gpu.
	Uses CUDA unified memory as staging buffer for uploading randomly positioned fine-grained tiles from host to device from much larger storage (RAM).
	*/
	template<typename T>
	struct TileManager {
		int deviceIndex;
		cudaStream_t stream;
		Helper::UnifiedMemory memoryForEncodedTiles;
		Helper::UnifiedMemory memoryForEncodedTrees;
		std::shared_ptr<std::mutex> tilesLock;
		std::shared_ptr<std::vector<HuffmanTileEncoder::Tile<T>>> tiles;
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		TileManager(T* terrainPtr, uint64_t width, uint64_t height, uint64_t tileWidth, uint64_t tileHeight,  int numThreads = std::thread::hardware_concurrency(), int deviceId = 0) {
			deviceIndex = deviceId;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			CUDA_CHECK(cudaStreamCreate(&stream));
			tiles = std::make_shared<std::vector<HuffmanTileEncoder::Tile<T>>>();
			tilesLock = std::make_shared<std::mutex>();
			int blockAlignedSize = HuffmanTileEncoder::computeBlockAlignedSize<T>(tileWidth, tileHeight);
			std::cout << "x = " << blockAlignedSize << std::endl;
			uint64_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint64_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint64_t numTiles = numTilesX * numTilesY;
			// Assuming encoded bits are not greater than raw data.
			memoryForEncodedTiles = Helper::UnifiedMemory(numTiles * blockAlignedSize);
			// Assuming maximum 511 nodes including internal nodes, 1 reserved for node count metadata.
			memoryForEncodedTrees = Helper::UnifiedMemory(numTiles * sizeof(uint16_t) * 512);
			std::cout << "Creating cpu workers..." << std::endl;
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(deviceIndex, i, terrainPtr, width, height, memoryForEncodedTiles, memoryForEncodedTrees, tilesLock);
				workers.push_back(worker);
			}
			std::cout << "Encoding tiles..." << std::endl;
			int idx = 0;
			for (uint64_t y = 0; y < numTilesY; y++) {
				for (uint64_t x = 0; x < numTilesX; x++) {
					int workerIndex = idx % numThreads;
					Helper::TileCommand<T> cmd;
					cmd.command = Helper::TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN;
					cmd.index = idx;
					cmd.tileSource.x1 = x * tileWidth;
					cmd.tileSource.y1 = y * tileHeight;
					cmd.tileSource.x2 = cmd.tileSource.x1 + tileWidth;
					cmd.tileSource.y2 = cmd.tileSource.y1 + tileHeight;
					workers[workerIndex]->addCommand(cmd);
					idx++;
				}
			}

			for (int i = 0; i < numThreads; i++) {
				workers[i]->wait();
			}
			std::cout << std::endl;
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTiles.ptr.get(), numTiles * blockAlignedSize, cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint16_t) * 512, cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamSynchronize(stream));

			// Test
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			void* args[] = { &tilePtr, &treePtr, &blockAlignedSize};
			CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_decodeTile, dim3(1, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}

		~TileManager() {
			workers.clear();
			std::unique_lock<std::mutex> lock(*tilesLock);
			CUDA_CHECK(cudaStreamDestroy(stream));
		}
	};
}
#endif