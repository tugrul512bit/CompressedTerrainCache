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
// For MSVC to see the header for syncthreads.
#define __CUDACC__
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
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.(stride of 256 within each tile)
		__global__ void k_decodeTile(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight);

		// Utilizes managed memory directly for all elements (has stride of terrain width)
		__global__ void k_accessTileNormally(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight);


		__global__ void k_accessDecodedTile(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight);

		__global__ void k_decodeSelectedTiles(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight,
			const uint32_t* tileIndexList);

		__global__ void k_accessSelectedTilesRawData(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight,
			const uint32_t* tileIndexList);

		__global__ void k_accessSelectedTiles(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight,
			const uint32_t* tileIndexList);
	}
	namespace Helper {
		struct UnifiedMemory {
			std::shared_ptr<unsigned char> ptr;
			uint32_t numBytes;
			UnifiedMemory(uint64_t sizeBytes = 0) {
				if (sizeBytes == 0) {
					ptr = nullptr;
					numBytes = 0;
				}
				else {
					numBytes = sizeBytes;
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
				CMD_MEASURE_HUFFMAN,
				CMD_ENCODE_HUFFMAN,
			};
			CMD command;
			HuffmanTileEncoder::Rectangle tileSource;
			int index;
			int blockAlignedTileBytes;
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
			int bitLengthMax;
			std::queue<TileCommand<T>> localCommandQueue;
			TileWorker(int deviceIndex, int index, T* terrainPtrPrm, uint64_t terrainWidth, uint64_t terrainHeight, 
				UnifiedMemory encodedTiles, 
				UnifiedMemory encodedTrees) 
				: commandQueue(), working(true), worker([&, index, terrainPtrPrm, encodedTiles, encodedTrees, terrainWidth, terrainHeight, deviceIndex]() {
				bool workingTmp = true;
				{
					std::unique_lock<std::mutex> lock(mutex);
					exiting = false;
					busy = true;
					bitLengthMax = 0;
				}
				if (terrainPtrPrm != nullptr) {

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
						int tmpBitLengthMax = 0;
						while (localCommandQueue.size() > 0) {
							TileCommand<T> task = localCommandQueue.front();
							localCommandQueue.pop();
							if (task.command == TileCommand<T>::CMD::CMD_MEASURE_HUFFMAN) {
								HuffmanTileEncoder::Tile<T> currentTile;
								currentTile.index = task.index;
								currentTile.area = task.tileSource;
								currentTile.copyInput(terrainWidth, terrainHeight, terrainPtrPrm);
								bool measureBitLength = true;
								int bitLength = currentTile.encode(measureBitLength);
								if (tmpBitLengthMax < bitLength) {
									tmpBitLengthMax = bitLength;
								}
							}
							if (task.command == TileCommand<T>::CMD::CMD_ENCODE_HUFFMAN) {
								HuffmanTileEncoder::Tile<T> currentTile;
								currentTile.index = task.index;
								currentTile.area = task.tileSource;
								currentTile.blockAlignedTileBytes = task.blockAlignedTileBytes;
								currentTile.copyInput(terrainWidth, terrainHeight, terrainPtrPrm);
								currentTile.encode();
								currentTile.copyOutput(encodedTiles.ptr.get(), encodedTrees.ptr.get());
							}
						}

						{
							std::unique_lock<std::mutex> lock(mutex);
							if (commandQueue.empty()) {
								busy = false;
								if (bitLengthMax < tmpBitLengthMax) {
									bitLengthMax = tmpBitLengthMax;
								}
							}
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
			int getMaxBitLength() {
				bool busyTmp = true;
				while (busyTmp) {
					{
						std::unique_lock<std::mutex> lock(mutex);
						busyTmp = busy;
					}
					cond.notify_one();
					std::this_thread::yield();
				}
				int result = 0;
				{
					std::unique_lock<std::mutex> lock(mutex);
					result = bitLengthMax;
					bitLengthMax = 0;
				}
				return result;
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
		Helper::UnifiedMemory memoryForOriginalTerrain;
		Helper::UnifiedMemory memoryForCustomBlockSelection;
		std::shared_ptr<std::mutex> tilesLock;
		std::shared_ptr<std::vector<HuffmanTileEncoder::Tile<T>>> tiles;
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		uint64_t tileWidth;
		uint64_t tileHeight;
		uint64_t width;
		uint64_t height;
		unsigned char* terrain;
		int blockAlignedTileBytes;
		uint32_t numBlocksToLaunch;
		TileManager(T* terrainPtr, uint64_t widthPrm, uint64_t heightPrm, uint64_t tileWidthPrm, uint64_t tileHeightPrm,  int numThreads = std::thread::hardware_concurrency(), int deviceId = 0) {
			tileWidth = tileWidthPrm;
			tileHeight = tileHeightPrm;
			width = widthPrm;
			height = heightPrm;
			deviceIndex = deviceId;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			CUDA_CHECK(cudaStreamCreate(&stream));
			cudaDeviceProp deviceProperties;
			CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceIndex));
			// Todo: optimize this for higher occupancy, without using non-in-flight-blocks.
			numBlocksToLaunch = deviceProperties.multiProcessorCount * 3;
			tiles = std::make_shared<std::vector<HuffmanTileEncoder::Tile<T>>>();

			uint64_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint64_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint64_t numTiles = numTilesX * numTilesY;
			
			// Assuming maximum 511 nodes including internal nodes, 1 reserved for node count metadata.
			memoryForEncodedTrees = Helper::UnifiedMemory(numTiles * sizeof(uint32_t) * 512);
			memoryForOriginalTerrain = Helper::UnifiedMemory(width * height * sizeof(T));
			terrain = memoryForOriginalTerrain.ptr.get();
			CUDA_CHECK(cudaMemcpyAsync(terrain, terrainPtr, width * height * sizeof(T), cudaMemcpyHostToDevice, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			std::cout << "Creating cpu workers..." << std::endl;
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(deviceIndex, i, terrainPtr, width, height, memoryForEncodedTiles, memoryForEncodedTrees);
				workers.push_back(worker);
			}
			std::cout << "Measuring encoding bitlengths of tiles..." << std::endl;
			int idx = 0;
			for (uint64_t y = 0; y < numTilesY; y++) {
				for (uint64_t x = 0; x < numTilesX; x++) {
					int workerIndex = idx % numThreads;
					Helper::TileCommand<T> cmd;
					cmd.command = Helper::TileCommand<T>::CMD::CMD_MEASURE_HUFFMAN;
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

			int maxBitLength = 0;
			for (int i = 0; i < numThreads; i++) {
				int currentMaxBitLength = workers[i]->getMaxBitLength();
				if (maxBitLength < currentMaxBitLength) {
					maxBitLength = currentMaxBitLength;
				}
			}
			std::cout << "max bit length per thread to decode per tile = " << maxBitLength << std::endl;
		
			// Calculating striped-decodable tile size.
			int num32BitChunksRequiredPerThread = (maxBitLength + sizeof(uint32_t) * 8 - 1) / (sizeof(uint32_t) * 8);
			blockAlignedTileBytes = sizeof(uint32_t) * num32BitChunksRequiredPerThread * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
			// Assuming encoded bits are not greater than raw data.
			memoryForEncodedTiles = Helper::UnifiedMemory(numTiles * blockAlignedTileBytes);
			for (int i = 0; i < numThreads; i++) {
				workers[i] = std::make_shared<Helper::TileWorker<T>>(deviceIndex, i, terrainPtr, width, height, memoryForEncodedTiles, memoryForEncodedTrees);
			}
			std::cout << "Encoding tiles...   block-aligned-tile-bytes="<< blockAlignedTileBytes << std::endl;
			idx = 0;
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
					cmd.blockAlignedTileBytes = blockAlignedTileBytes;
					workers[workerIndex]->addCommand(cmd);
					idx++;
				}
			}
			
			for (int i = 0; i < numThreads; i++) {
				workers[i]->wait();
			}
			
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTiles.ptr.get(), numTiles * blockAlignedTileBytes, cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamSynchronize(stream));

			int concurrentManagedAccess = 0;
			CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess,cudaDevAttrConcurrentManagedAccess, deviceIndex));
			std::cout << "Device support for concurrent managed access: " << concurrentManagedAccess << std::endl;
			if (concurrentManagedAccess != 0) {
				cudaMemLocation loc;
				loc.id = deviceIndex;
				loc.type = cudaMemLocationTypeDevice;
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTiles.ptr.get(), numTiles * blockAlignedTileBytes, cudaMemAdviseSetAccessedBy, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAdviseSetAccessedBy, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAdviseSetAccessedBy, loc));
			}
			else {
				// Fallback to read-mostly that doesn't avoid uncached reads.
				std::cout << "Concurrent managed access not supported, falling back to cudaMemAdviseSetReadMostly mode." << std::endl;
				cudaMemLocation loc;
				loc.id = deviceIndex;
				loc.type = cudaMemLocationTypeDevice;
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTiles.ptr.get(), numTiles * blockAlignedTileBytes, cudaMemAdviseSetReadMostly, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAdviseSetReadMostly, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAdviseSetReadMostly, loc));
			}
		}
		void unitTestForDataIntegrity() {
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint32_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint32_t numTiles = numTilesX * numTilesY;
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th };
			CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_decodeTile, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}
		void benchmarkNormalAccess() {
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint32_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint32_t numTiles = numTilesX * numTilesY;
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th };
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 20; i++) {
				CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_accessTileNormally, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			auto time = (dur.count() / 20000000.0);
			double dataSize = tileSizeBytes * (double)numTiles;
			double throughput = dataSize / time;
			std::cout << "Accessing " << numTiles << " tiles of raw terrain data through unified memory(RAM->VRAM): " << time << " seconds per kernel.Throughput = " << throughput / (1024 * 1024 * 1024) << " GB / s " << std::endl;
		}
		void benchmarkDecodedAccess() {
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint32_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint32_t numTiles = numTilesX * numTilesY;
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th };
			auto start = std::chrono::high_resolution_clock::now();
			for (int i = 0; i < 20; i++) {
				CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_accessDecodedTile, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			auto time = (dur.count() / 20000000.0);
			double dataSize = tileSizeBytes * (double)numTiles;
			double throughput = dataSize / time;
			std::cout << "Accessing " << numTiles << " encoded tiles through unified memory (RAM -> VRAM) + decoding: " << time << " seconds per kernel. Throughput = " << throughput / (1024 * 1024 * 1024) << " GB/s " << std::endl;
		}
		void unitTestForSelectedDataIntegrity(std::vector<uint32_t> tileIndexList) {
			uint32_t selectionBytes = tileIndexList.size() * sizeof(uint32_t);
			if (memoryForCustomBlockSelection.numBytes < selectionBytes) {
				memoryForCustomBlockSelection = Helper::UnifiedMemory(selectionBytes);
			}
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTiles = tileIndexList.size();
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			CUDA_CHECK(cudaMemcpyAsync(memoryForCustomBlockSelection.ptr.get(), tileIndexList.data(), selectionBytes, cudaMemcpyHostToDevice, stream));
			uint32_t* tileList = reinterpret_cast<uint32_t*>(memoryForCustomBlockSelection.ptr.get());
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th, &tileList };
			CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_decodeSelectedTiles, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
		}
		void benchmarkForSelectedTilesNormalAccess(std::vector<uint32_t> tileIndexList) {
			uint32_t selectionBytes = tileIndexList.size() * sizeof(uint32_t);
			if (memoryForCustomBlockSelection.numBytes < selectionBytes) {
				memoryForCustomBlockSelection = Helper::UnifiedMemory(selectionBytes);
			}
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTiles = tileIndexList.size();
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			auto start = std::chrono::high_resolution_clock::now();
			CUDA_CHECK(cudaMemcpyAsync(memoryForCustomBlockSelection.ptr.get(), tileIndexList.data(), selectionBytes, cudaMemcpyHostToDevice, stream));
			uint32_t* tileList = reinterpret_cast<uint32_t*>(memoryForCustomBlockSelection.ptr.get());
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th, &tileList };
			for (int i = 0; i < 20; i++) {
				CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_accessSelectedTilesRawData, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			auto time = (dur.count() / 20000000.0);
			double dataSize = tileSizeBytes * (double)numTiles;
			double throughput = dataSize / time;
			std::cout << "Accessing " << numTiles << " tiles of raw terrain data through unified memory(RAM->VRAM): " << time << " seconds per kernel.Throughput = " << throughput / (1024 * 1024 * 1024) << " GB / s " << std::endl;
		}
		void benchmarkForSelectedTilesEncodedAccess(std::vector<uint32_t> tileIndexList) {
			uint32_t selectionBytes = tileIndexList.size() * sizeof(uint32_t);
			if (memoryForCustomBlockSelection.numBytes < selectionBytes) {
				memoryForCustomBlockSelection = Helper::UnifiedMemory(selectionBytes);
			}
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t numTiles = tileIndexList.size();
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			auto start = std::chrono::high_resolution_clock::now();
			CUDA_CHECK(cudaMemcpyAsync(memoryForCustomBlockSelection.ptr.get(), tileIndexList.data(), selectionBytes, cudaMemcpyHostToDevice, stream));
			uint32_t* tileList = reinterpret_cast<uint32_t*>(memoryForCustomBlockSelection.ptr.get());
			void* args[] = { &tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th, &tileList };
			for (int i = 0; i < 20; i++) {
				CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_accessSelectedTiles, dim3(numBlocksToLaunch, 1, 1), dim3(HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK, 1, 1), args, 0, stream));
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
			auto end = std::chrono::high_resolution_clock::now();
			auto dur = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
			auto time = (dur.count() / 20000000.0);
			double dataSize = tileSizeBytes * (double)numTiles;
			double throughput = dataSize / time;
			std::cout << "Accessing "<< numTiles <<" encoded tiles through unified memory(RAM->VRAM) + decoding: " << time << " seconds per kernel.Throughput = " << throughput / (1024 * 1024 * 1024) << " GB / s " << std::endl;
		}
		~TileManager() {
			workers.clear();
			CUDA_CHECK(cudaStreamDestroy(stream));
		}
	};
}
#endif