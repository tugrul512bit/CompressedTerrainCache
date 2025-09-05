#ifndef _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#define _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#include <vector>
#include <thread>
#include <memory>
#include <queue>
#include <mutex>
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
		template<typename T>
		__global__ void k_accessSelectedTiles(	const uint32_t tileSizeBytes,
												const unsigned char* originalTileDataForComparison_u,
												const uint32_t numTilesToTest,
												const uint32_t terrainWidth,
												const uint32_t terrainHeight,
												const uint32_t tileWidth,
												const uint32_t tileHeight,
												const uint32_t* tileIndexList_u,
												unsigned char* outputTiles_d) {
			constexpr uint32_t numBytesPerElement = sizeof(T);
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					const uint32_t tile = tileIndexList_u[tileIndex];
					const uint32_t numAccessSteps = (tileSizeBytes + blockDim.x - 1) / blockDim.x;
					// Decode steps.
					uint32_t chunkCache = 0;
					uint32_t chunkCacheIndex = -1;
					for (uint32_t decodeStep = 0; decodeStep < numAccessSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * blockDim.x + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							const uint32_t tileLocalX = (byteIndex / numBytesPerElement) % tileWidth;
							const uint32_t tileLocalY = (byteIndex / numBytesPerElement) / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) {
								outputTiles_d[tileIndex * (uint64_t)tileSizeBytes + byteIndex] = originalTileDataForComparison_u[globalX * numBytesPerElement + globalY * terrainWidth * numBytesPerElement + (byteIndex % numBytesPerElement)];
							}
						}
					}
				}
			}
		}
		__device__ __forceinline__ bool d_acquireDirectMappedCacheSlot(uint32_t tile, uint32_t numTilesX, uint32_t numTilesY, uint32_t numSlotsX, uint32_t numSlotsY, uint32_t* slotLocks_d, uint32_t* tileCacheDataIndex_d, uint32_t localThreadIndex, uint32_t* s_broadcast, uint32_t& cacheSlotIndexOut) {
			const uint32_t tileX = tile % numTilesX;
			const uint32_t tileY = tile / numTilesX;
			uint32_t slotIndexX = tileX % numSlotsX;
			uint32_t slotIndexY = tileY % numSlotsY;
			uint32_t slotIndex = slotIndexX + slotIndexY * numSlotsX;
			bool cacheHit = false;
			// Block leader locks the slot and the block waits for the leader.
			if (localThreadIndex == 0) {
				uint32_t exponentialBackoff = 1;
				while (atomicCAS(&slotLocks_d[slotIndex], 0, 1) != 0) {
					__nanosleep(exponentialBackoff);
					if (exponentialBackoff < 1024 * 8) {
						exponentialBackoff = exponentialBackoff << 1;
					}
				}
				__threadfence();
				// if cache-miss, enable decoding and loading for the data from unified memory.
				if (tileCacheDataIndex_d[slotIndex] != tile) {
					// Mark new tile for this slot.
					tileCacheDataIndex_d[slotIndex] = tile;
					s_broadcast[0] = 1;
				}
				else {
					s_broadcast[0] = 0;
				}
			}
			__syncthreads();
			cacheHit = (0 == s_broadcast[0]);
			cacheSlotIndexOut = slotIndex;
			return cacheHit;
		}

		__device__ __forceinline__ void d_releaseDirectMappedCacheSlot(uint32_t slotIndex, uint32_t* slotLocks_d, uint32_t localThreadIndex) {
			__syncthreads();
			// Block leader locks the slot and the block waits for the leader.
			if (localThreadIndex == 0) {
				__threadfence();
				atomicExch(&slotLocks_d[slotIndex], 0);
			}
		}

		template<typename T>
		__global__ void k_decodeSelectedTilesWithDirectMappedCache(	const unsigned char* encodedTiles_u,
																	const unsigned char* encodedTrees_u,
																	const uint32_t blockAlignedElements,
																	const uint32_t tileSizeBytes,
																	const uint32_t numTilesToTest,
																	const uint32_t terrainWidth,
																	const uint32_t terrainHeight,
																	const uint32_t tileWidth,
																	const uint32_t tileHeight,
																	const uint32_t* tileIndexList_u,
																	unsigned char* outputTiles_d,
																	uint32_t* tileCacheSlotLock_d,
																	const uint32_t numTileCacheSlotsX,
																	const uint32_t numTileCacheSlotsY,
																	uint32_t* tileCacheDataIndex_d,
																	unsigned char* cache_d) {
			constexpr uint32_t numBytesPerElement = sizeof(T);
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t* treePtr_u = reinterpret_cast<const uint32_t*>(encodedTrees_u);
			const uint32_t* tilePtr_u = reinterpret_cast<const uint32_t*>(encodedTiles_u);
			__shared__ uint32_t s_tree[511];
			// Todo: alignas(sizeof(T)) !!!
			extern __shared__ unsigned char s_coalescingLayer[];
			__shared__ uint32_t s_broadcast[1];
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			const uint32_t numDecodeSteps = (tileSizeBytes + blockDim.x - 1) / blockDim.x;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					const uint32_t tile = tileIndexList_u[tileIndex];
					if (tileIndex + numBlocks < numTilesToTest) {
						asm("prefetch.global.L1 [%0];"::"l"(&tileIndexList_u[tileIndex + numBlocks]));
					}
					// Acquiring cache slot and computing cache-hit or cache-miss.
					uint32_t cacheSlotIndexOut = 0;
					bool cacheHit = d_acquireDirectMappedCacheSlot(tile, numTilesX, numTilesY, numTileCacheSlotsX, numTileCacheSlotsY, tileCacheSlotLock_d, tileCacheDataIndex_d, localThreadIndex, &s_broadcast[0], cacheSlotIndexOut);
					const uint64_t cacheSlotOffset = cacheSlotIndexOut * (uint64_t)tileSizeBytes;
					// Cache-hit (uses VRAM cache as source)
					if (cacheHit) {
						const uint64_t destinationOffset = tileIndex * (uint64_t)tileSizeBytes;
						const uint32_t numCopySteps = (tileSizeBytes + blockDim.x * numBytesPerElement - 1) / (blockDim.x * numBytesPerElement);
						for (uint32_t copyStep = 0; copyStep < numCopySteps; copyStep++) {
							const uint32_t tIndex = copyStep * blockDim.x * numBytesPerElement + localThreadIndex * numBytesPerElement;
							if (tIndex < tileSizeBytes) {
								*reinterpret_cast<T*>(&outputTiles_d[destinationOffset + tIndex]) = *reinterpret_cast<T*>(&cache_d[cacheSlotOffset + tIndex]);
							}
						}
						d_releaseDirectMappedCacheSlot(cacheSlotIndexOut, tileCacheSlotLock_d, localThreadIndex);
						continue;
					}
					// Cache-miss step 1: streams data from RAM, decodes the data and writes to device memory output
					// Cache-miss step 2: copy the decoded data to the cache slot.
					uint32_t decodeBitIndex = 0;
					constexpr uint32_t one = 1;
					const uint32_t* chunkBlockPtr_u = &tilePtr_u[blockAlignedElements * (uint64_t)tile];
					const uint32_t* treeBlockPtr_u = &treePtr_u[512 * tile];
					// Loading tree into smem.
					const uint32_t numNodes = treeBlockPtr_u[0];
					const uint32_t numTreeLoadingSteps = (numNodes + blockDim.x - 1) / blockDim.x;
					for (int l = 0; l < numTreeLoadingSteps; l++) {
						uint32_t node = l * blockDim.x + localThreadIndex;
						if (node < numNodes) {
							s_tree[node] = treeBlockPtr_u[1 + node];
						}
					}
					__syncthreads();
					// Decode steps.
					uint32_t chunkCache = 0;
					uint32_t chunkCacheIndex = -1;
					for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * blockDim.x + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							unsigned char leafNodeFound = 0;
							uint32_t currentNodeIndex = 0;
							uint8_t symbol = 0;
							while (!leafNodeFound) {
								const uint32_t chunkColumn = localThreadIndex;
								const uint32_t chunkRow = decodeBitIndex >> 5;
								const uint32_t chunkBit = decodeBitIndex & 31;
								// Aggregated access to the unified mem.
								const uint32_t chunkLoadIndex = chunkColumn + chunkRow * blockDim.x;
								if (chunkCacheIndex != chunkLoadIndex) {
									chunkCache = chunkBlockPtr_u[chunkLoadIndex];
									chunkCacheIndex = chunkLoadIndex;
									if (chunkLoadIndex + blockDim.x < blockAlignedElements) {
										asm("prefetch.global.L1 [%0];"::"l"(&chunkBlockPtr_u[chunkLoadIndex + blockDim.x]));
									}
								}
								const uint32_t bitBeingDecoded = (chunkCache >> chunkBit) & one;
								const uint32_t node = s_tree[currentNodeIndex];
								leafNodeFound = (node >> 8) & 0b11111111;
								const uint16_t childNodeStart = node >> 16;
								symbol = node & 0b11111111;
								currentNodeIndex = bitBeingDecoded ? childNodeStart + 1 : childNodeStart;
								decodeBitIndex++;
							}
							decodeBitIndex--;
							s_coalescingLayer[localThreadIndex] = symbol;
						}
						else {
							s_coalescingLayer[localThreadIndex] = 0;
						}
						__syncthreads();
						uint32_t blockByteOffset = byteIndex - localThreadIndex;
						uint32_t localThreadOffset = localThreadIndex * numBytesPerElement;
						uint32_t writeIndex = blockByteOffset + localThreadOffset;
						if (writeIndex < tileSizeBytes) {
							const T obj = *reinterpret_cast<T*>(&s_coalescingLayer[localThreadOffset]);
							// Copying to the output.
							*reinterpret_cast<T*>(&outputTiles_d[tileIndex * (uint64_t)tileSizeBytes + writeIndex]) = obj;
							// Updating the cache.
							*reinterpret_cast<T*>(&cache_d[cacheSlotOffset + writeIndex]) = obj;
						}
						__syncthreads();
					}
					d_releaseDirectMappedCacheSlot(cacheSlotIndexOut, tileCacheSlotLock_d, localThreadIndex);
				}
			}
		}
	}
	namespace Helper {
		struct DeviceMemory {
			std::shared_ptr<unsigned char> ptr;
			uint64_t numBytes;
			DeviceMemory(cudaStream_t stream = cudaStream_t(), uint64_t sizeBytes = 0) {
				if (sizeBytes == 0) {
					ptr = nullptr;
					numBytes = 0;
				}
				else {
					numBytes = sizeBytes;
					if (numBytes > 0) {
						unsigned char* tmp;
						CUDA_CHECK(cudaMallocAsync(&tmp, sizeBytes, stream));
						CUDA_CHECK(cudaStreamSynchronize(stream));
						ptr = std::shared_ptr<unsigned char>(tmp, [stream, sizeBytes](unsigned char* ptr) { 
							if (sizeBytes > 0) { 
								CUDA_CHECK(cudaFreeAsync(ptr, stream));
								CUDA_CHECK(cudaStreamSynchronize(stream)); 
							}
						});
					}
				}
			}
			~DeviceMemory() {}
		};
		struct UnifiedMemory {
			std::shared_ptr<unsigned char> ptr;
			uint64_t numBytes;
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
			~UnifiedMemory() {}
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
				UnifiedMemory encodedTrees,
				uint32_t cudaBlockSize) 
				: commandQueue(), working(true), worker([&,cudaBlockSize, index, terrainPtrPrm, encodedTiles, encodedTrees, terrainWidth, terrainHeight, deviceIndex]() {
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
								int bitLength = currentTile.encode(measureBitLength, cudaBlockSize);
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
								currentTile.encode(false, cudaBlockSize);
								currentTile.copyOutput(encodedTiles.ptr.get(), encodedTrees.ptr.get());
							}
						}

						{
							std::unique_lock<std::mutex> lock(mutex);
							if (commandQueue.empty()) {
								busy = false;
							}
							if (bitLengthMax < tmpBitLengthMax) {
								bitLengthMax = tmpBitLengthMax;
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
	Produces Huffman-Encoded tiles from a 2D terrain. When accessing from device, it checks if 2D direct-mapped-cache contains the tile, then returns it directly if it exists in device memory. 
	In case of a cache-miss, it transfers encoded tile data and decodes to fill cache slot and use the data.
	After the data is ready, it passes through the CUDA-compressible-memory to increase chance of hardware cache-hit when accessing dataset larger than L2 cache of gpu.
	*/
	template<typename T>
	struct TileManager {
		int deviceIndex;
		cudaStream_t stream;
		cudaEvent_t start, stop;
		// This is for the result or output to be used in a game logic that needs surroundings on a terrain data directly from device-memory.
		Helper::DeviceMemory memoryForLoadedTerrain;
		// This is for the encoded tiles to be used in efficient data retrieval from host to device for any unknown pattern in runtime.
		Helper::UnifiedMemory memoryForEncodedTiles;
		// This is for the Huffman Tree of each encoded tile, maximum 512 integers (8bit symbols).
		Helper::UnifiedMemory memoryForEncodedTrees;
		// This is for the original terrain data to be used as comparisons in benchmarks or data-integrity checks.
		Helper::UnifiedMemory memoryForOriginalTerrain;
		// This is an input for selecting tiles dynamically from host-given array of tile indices (row-major tile order)
		Helper::UnifiedMemory memoryForCustomBlockSelection;
		// Array of locks for cache slots in device-memory.
		Helper::DeviceMemory memoryForTileCacheSlotLock;
		// Size of cache in terms of tiles.
		uint32_t numTileCacheSlotsX;
		uint32_t numTileCacheSlotsY;
		// Array of tile indices for cache slots.
		Helper::DeviceMemory memoryForTileCacheDataIndex;
		// Cache data.
		Helper::DeviceMemory memoryForCache;
		std::vector<std::shared_ptr<Helper::TileWorker<T>>> workers;
		uint64_t tileWidth;
		uint64_t tileHeight;
		uint64_t width;
		uint64_t height;
		unsigned char* terrain;
		int blockAlignedTileBytes;
		bool benchmarkComparisonEnabled;
		int cudaGridSize;
		int cudaBlockSize;
		/* 
		T: type of units in the terrain (currently only unsigned char supported).
		terrainPtr: raw data pointer to terrain data on host.
		widthPrm: width of terrain in units
		heightPrm: height of terrain in units
		tileWidthPrm: width of each tile in units
		tileHeightPrm: height of each tile in units
		numThreads: number of cpu threads to use in construction of linearized tiles from terrain data
		deviceId: CUDA device index to be used for CUDA kernels and memory allocations.
		allocateExtraTerrainForBenchmarking: true = enables a copy of raw terrain data in unified memory to be compared against encoding method (doubles memory requirement)
		*/
		TileManager(T* terrainPtr, uint64_t widthPrm, uint64_t heightPrm, 
					uint64_t tileWidthPrm, uint64_t tileHeightPrm,	
					uint64_t tileCacheSlotColumns, uint64_t tileCacheSlotRows,
					int numThreads = std::thread::hardware_concurrency(), 
					int deviceId = 0, bool allocateExtraTerrainForBenchmarking = true) {
			tileWidth = tileWidthPrm;
			tileHeight = tileHeightPrm;
			width = widthPrm;
			height = heightPrm;
			deviceIndex = deviceId;
			benchmarkComparisonEnabled = allocateExtraTerrainForBenchmarking;
			CUDA_CHECK(cudaInitDevice(deviceIndex, cudaDeviceScheduleAuto, cudaInitDeviceFlagsAreValid));
			CUDA_CHECK(cudaSetDevice(deviceIndex));
			CUDA_CHECK(cudaStreamCreate(&stream));
			CUDA_CHECK(cudaEventCreate(&start));
			CUDA_CHECK(cudaEventCreate(&stop));
			CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&cudaGridSize, &cudaBlockSize, (void*)Kernels::k_decodeSelectedTilesWithDirectMappedCache<T>, sizeof(T) * cudaBlockSize + sizeof(T)));
			
			// Allocating cache in device memory.
			numTileCacheSlotsX = tileCacheSlotColumns;
			numTileCacheSlotsY = tileCacheSlotRows;
			// Array of locks for cache slots in device-memory.
			memoryForTileCacheSlotLock = Helper::DeviceMemory(stream, sizeof(uint32_t) * numTileCacheSlotsX * numTileCacheSlotsY);
			CUDA_CHECK(cudaMemsetAsync(memoryForTileCacheSlotLock.ptr.get(), 0, memoryForTileCacheSlotLock.numBytes, stream));
			// Array of tile indices for cache slots.
			memoryForTileCacheDataIndex = Helper::DeviceMemory(stream, sizeof(uint32_t) * numTileCacheSlotsX * numTileCacheSlotsY);
			CUDA_CHECK(cudaMemsetAsync(memoryForTileCacheDataIndex.ptr.get(), 255, memoryForTileCacheDataIndex.numBytes, stream));
			// Cache data.
			memoryForCache = Helper::DeviceMemory(stream, sizeof(T) * tileWidth * tileHeight * numTileCacheSlotsX * numTileCacheSlotsY);

			uint64_t numTilesX = (width + tileWidth - 1) / tileWidth;
			uint64_t numTilesY = (height + tileHeight - 1) / tileHeight;
			uint64_t numTiles = numTilesX * numTilesY;

			// Assuming maximum 511 nodes including internal nodes, 1 reserved for node count metadata.
			memoryForEncodedTrees = Helper::UnifiedMemory(numTiles * sizeof(uint32_t) * 512);
			if (benchmarkComparisonEnabled) {
				memoryForOriginalTerrain = Helper::UnifiedMemory(width * height * sizeof(T));
				terrain = memoryForOriginalTerrain.ptr.get();
				CUDA_CHECK(cudaMemcpyAsync(terrain, terrainPtr, width * height * sizeof(T), cudaMemcpyHostToDevice, stream));
				CUDA_CHECK(cudaStreamSynchronize(stream));
			}
			std::cout << "Creating cpu workers..." << std::endl;
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<Helper::TileWorker<T>> worker = std::make_shared<Helper::TileWorker<T>>(deviceIndex, i, terrainPtr, width, height, memoryForEncodedTiles, memoryForEncodedTrees, cudaBlockSize);
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
			uint32_t num32BitChunksRequiredPerThread = (maxBitLength + sizeof(uint32_t) * 8 + 1) / (sizeof(uint32_t) * 8);
			blockAlignedTileBytes = sizeof(uint32_t) * num32BitChunksRequiredPerThread * cudaBlockSize;
			// Assuming encoded bits are not greater than raw data.
			memoryForEncodedTiles = Helper::UnifiedMemory(numTiles * (uint64_t) blockAlignedTileBytes);
			for (int i = 0; i < numThreads; i++) {
				workers[i] = std::make_shared<Helper::TileWorker<T>>(deviceIndex, i, terrainPtr, width, height, memoryForEncodedTiles, memoryForEncodedTrees, cudaBlockSize);
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
			
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTiles.ptr.get(), numTiles * (uint64_t)blockAlignedTileBytes, cudaMemAttachGlobal));
			CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAttachGlobal));
			if (benchmarkComparisonEnabled) {
				CUDA_CHECK(cudaStreamAttachMemAsync(stream, memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAttachGlobal));
			}
			CUDA_CHECK(cudaStreamSynchronize(stream));

			int concurrentManagedAccess = 0;
			CUDA_CHECK(cudaDeviceGetAttribute(&concurrentManagedAccess,cudaDevAttrConcurrentManagedAccess, deviceIndex));
			std::cout << "Device support for concurrent managed access: " << concurrentManagedAccess << std::endl;
			if (concurrentManagedAccess != 0) {
				cudaMemLocation loc;
				loc.id = deviceIndex;
				loc.type = cudaMemLocationTypeDevice;
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTiles.ptr.get(), numTiles * (uint64_t)blockAlignedTileBytes, cudaMemAdviseSetAccessedBy, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAdviseSetAccessedBy, loc));
				if (benchmarkComparisonEnabled) {
					CUDA_CHECK(cudaMemAdvise(memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAdviseSetAccessedBy, loc));
				}
			}
			else {
				// Fallback to read-mostly that doesn't avoid uncached reads.
				std::cout << "Concurrent managed access not supported, falling back to cudaMemAdviseSetReadMostly mode." << std::endl;
				cudaMemLocation loc;
				loc.id = deviceIndex;
				loc.type = cudaMemLocationTypeDevice;
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTiles.ptr.get(), numTiles * (uint64_t)blockAlignedTileBytes, cudaMemAdviseSetReadMostly, loc));
				CUDA_CHECK(cudaMemAdvise(memoryForEncodedTrees.ptr.get(), numTiles * sizeof(uint32_t) * 512, cudaMemAdviseSetReadMostly, loc));
				if (benchmarkComparisonEnabled) {
					CUDA_CHECK(cudaMemAdvise(memoryForOriginalTerrain.ptr.get(), width * height * sizeof(T), cudaMemAdviseSetReadMostly, loc));
				}
			}
		}
		
		/* 
		Directly uses unified memory to fetch tile data.
		Takes row-major list of tile indices to stream from RAM to VRAM.
		Returns tiles loaded in the same order as the indices.
		Elapsed time (in seconds) is written to the last parameter.
		*/
		unsigned char* accessSelectedTiles(std::vector<uint32_t> tileIndexList, double* elapsedTime = nullptr, double* dataSize = nullptr, double* throughput = nullptr) {
			if (!benchmarkComparisonEnabled) {
				std::cout << "Error: allocation for benchmarking against normal method is not enabled. Construct CompressedTerrainCache with allocateExtraTerrainForBenchmarking = true for the extra allocation." << std::endl;
				return nullptr;
			}
			uint32_t numTiles = tileIndexList.size();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t selectionBytes = tileIndexList.size() * sizeof(uint32_t);
			if (memoryForCustomBlockSelection.numBytes < selectionBytes) {
				memoryForCustomBlockSelection = Helper::UnifiedMemory(selectionBytes);
			}
			if (memoryForLoadedTerrain.numBytes < tileSizeBytes * (uint64_t)numTiles) {
				memoryForLoadedTerrain = Helper::DeviceMemory(stream, tileSizeBytes * (uint64_t)numTiles);
			}

			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			CUDA_CHECK(cudaMemcpyAsync(memoryForCustomBlockSelection.ptr.get(), tileIndexList.data(), selectionBytes, cudaMemcpyHostToDevice, stream));
			uint32_t* tileList_u = reinterpret_cast<uint32_t*>(memoryForCustomBlockSelection.ptr.get());
			unsigned char* output_d = memoryForLoadedTerrain.ptr.get();
			void* args[] = { &tileSizeBytes, &terrain, &numTiles, &w, &h, &tw, &th, &tileList_u, &output_d };
			CUDA_CHECK(cudaEventRecord(start, stream));
			CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_accessSelectedTiles<T>, dim3(cudaGridSize, 1, 1), dim3(cudaBlockSize, 1, 1), args, 0, stream));
			CUDA_CHECK(cudaEventRecord(stop, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			float milliseconds = 0.0f;
			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
			double time = milliseconds / 1000.0;
			if (dataSize) {
				*dataSize = (tileSizeBytes * (double)numTiles) / 1000000000.0;
			}
			if (throughput) {
				*throughput = *dataSize / time;
			}
			if (elapsedTime) {
				*elapsedTime = time;
			}
			return memoryForLoadedTerrain.ptr.get();
		}
		// Fetches Huffman Tree, encoded linearized tile data and decodes multiple columns in parallel.
		unsigned char* decodeSelectedTiles(std::vector<uint32_t> tileIndexList, double* elapsedTime = nullptr, double* dataSize = nullptr, double* throughput = nullptr) {
			uint32_t numTiles = tileIndexList.size();
			uint32_t tileSizeBytes = tileWidth * tileHeight * sizeof(T);
			uint32_t selectionBytes = tileIndexList.size() * sizeof(uint32_t);
			if (memoryForCustomBlockSelection.numBytes < selectionBytes) {
				memoryForCustomBlockSelection = Helper::UnifiedMemory(selectionBytes);
			}
			if (memoryForLoadedTerrain.numBytes < tileSizeBytes * (uint64_t)numTiles) {
				memoryForLoadedTerrain = Helper::DeviceMemory(stream, tileSizeBytes * (uint64_t)numTiles);
			}
			uint32_t blockAligned32BitElements = blockAlignedTileBytes / sizeof(uint32_t);
			unsigned char* tilePtr = memoryForEncodedTiles.ptr.get();
			unsigned char* treePtr = memoryForEncodedTrees.ptr.get();
			uint32_t w = width;
			uint32_t h = height;
			uint32_t tw = tileWidth;
			uint32_t th = tileHeight;
			CUDA_CHECK(cudaMemcpyAsync(memoryForCustomBlockSelection.ptr.get(), tileIndexList.data(), selectionBytes, cudaMemcpyHostToDevice, stream));
			uint32_t* tileList_u = reinterpret_cast<uint32_t*>(memoryForCustomBlockSelection.ptr.get());
			unsigned char* output = memoryForLoadedTerrain.ptr.get();
			
			uint32_t* tileCacheSlotLock_d = reinterpret_cast<uint32_t*>(memoryForTileCacheSlotLock.ptr.get());
			uint32_t* tileCacheDataIndex_d = reinterpret_cast<uint32_t*>(memoryForTileCacheDataIndex.ptr.get());
			unsigned char* cache_d = memoryForCache.ptr.get();

			void* args[] = { 
				&tilePtr, &treePtr, &blockAligned32BitElements, &tileSizeBytes, &numTiles, &w, &h, &tw, &th, & tileList_u, &output,
				&tileCacheSlotLock_d, &numTileCacheSlotsX, &numTileCacheSlotsY, &tileCacheDataIndex_d, &cache_d
			};
			CUDA_CHECK(cudaEventRecord(start, stream));
			CUDA_CHECK(cudaLaunchKernel((void*)Kernels::k_decodeSelectedTilesWithDirectMappedCache<T>, dim3(cudaGridSize, 1, 1), dim3(cudaBlockSize, 1, 1), args, sizeof(T) * cudaBlockSize + sizeof(T), stream));
			CUDA_CHECK(cudaEventRecord(stop, stream));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			float milliseconds = 0.0f;
			CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
			double time = milliseconds / 1000.0;
			if (dataSize) {
				*dataSize = (tileSizeBytes * (double)numTiles) / 1000000000.0;
			}
			if (throughput) {
				*throughput = *dataSize / time;
			}
			if (elapsedTime) {
				*elapsedTime = time;
			}
			return memoryForLoadedTerrain.ptr.get();
		}
		~TileManager() {
			workers.clear();
			memoryForLoadedTerrain.~DeviceMemory();
			memoryForTileCacheSlotLock.~DeviceMemory();
			memoryForTileCacheDataIndex.~DeviceMemory();
			memoryForCache.~DeviceMemory();
			CUDA_CHECK(cudaEventDestroy(start));
			CUDA_CHECK(cudaEventDestroy(stop));
			CUDA_CHECK(cudaStreamSynchronize(stream));
			CUDA_CHECK(cudaStreamDestroy(stream));
		}
	};
}
#endif