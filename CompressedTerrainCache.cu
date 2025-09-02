#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {

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
			const uint32_t* tileIndexList,
			unsigned char* outputTiles) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					const uint32_t tile = tileIndexList[tileIndex];
					const uint32_t numAccessSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					// Decode steps.
					uint32_t chunkCache = 0;
					uint32_t chunkCacheIndex = -1;
					for (uint32_t decodeStep = 0; decodeStep < numAccessSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) { 
								outputTiles[tileIndex * (uint64_t)tileSizeBytes + byteIndex] = originalTileDataForComparison[globalX + globalY * terrainWidth];
							}
						}
					}
				}
			}
		}

		__device__ __forceinline__ bool d_acquireDirectMappedCacheSlot(uint32_t tile, uint32_t numTilesX, uint32_t numTilesY, uint32_t numSlotsX, uint32_t numSlotsY, uint32_t* slotLocks, uint32_t* tileCacheDataIndex, uint32_t localThreadIndex, uint32_t* broadcast, uint32_t& cacheSlotIndexOut) {
			const uint32_t tileX = tile % numTilesX;
			const uint32_t tileY = tile / numTilesY;
			uint32_t slotIndexX = tileX % numSlotsX;
			uint32_t slotIndexY = tileY % numSlotsY;
			uint32_t slotIndex = slotIndexX + slotIndexY * numSlotsX;
			bool cacheHit = false;
			// Block leader locks the slot and the block waits for the leader.
			if (localThreadIndex == 0) {
				uint32_t exponentialBackoff = 4;
				while (atomicCAS(&slotLocks[slotIndex], 0, 1) != 0) {
					__nanosleep(exponentialBackoff);
					if (exponentialBackoff < 1024 * 8) {
						exponentialBackoff = exponentialBackoff << 1;
					}
				}
				__threadfence();
				// if cache-miss, enable decoding and loading for the data from unified memory.
				if (tileCacheDataIndex[slotIndex] != tile) {
					// Mark new tile for this slot.
					tileCacheDataIndex[slotIndex] = tile;
					broadcast[0] = 1;
				}
				else {
					broadcast[0] = 0;
				}
			}
			__syncthreads();
			cacheHit = (0 == broadcast[0]);
			cacheSlotIndexOut = slotIndex;
			return cacheHit;
		}

		__device__ __forceinline__ void d_releaseDirectMappedCacheSlot(uint32_t slotIndex, uint32_t* slotLocks, uint32_t localThreadIndex) {
			__syncthreads();
			// Block leader locks the slot and the block waits for the leader.
			if (localThreadIndex == 0) {
				__threadfence();
				atomicExch(&slotLocks[slotIndex], 0);
			}
		}

		__global__ void k_decodeSelectedTilesWithDirectMappedCache(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight,
			const uint32_t* tileIndexList,
			unsigned char* outputTiles,
			uint32_t* tileCacheSlotLock,
			const uint32_t numTileCacheSlotsX,
			const uint32_t numTileCacheSlotsY,
			uint32_t* tileCacheDataIndex,
			unsigned char* cache) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			extern __shared__ uint32_t s_tree[];
			__shared__ uint32_t s_broadcast[1];
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			const uint32_t numDecodeSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					const uint32_t tile = tileIndexList[tileIndex];

					// Acquiring cache slot and computing cache-hit or cache-miss.
					uint32_t cacheSlotIndexOut = 0;
					bool cacheHit = d_acquireDirectMappedCacheSlot(tile, numTilesX, numTilesY, numTileCacheSlotsX, numTileCacheSlotsY, tileCacheSlotLock, tileCacheDataIndex, localThreadIndex, &s_broadcast[0], cacheSlotIndexOut);
					const uint64_t sourceOffset = cacheSlotIndexOut * (uint64_t)tileSizeBytes;
					
					// Cache-hit (uses VRAM cache as source)
					if (cacheHit) {
						
						const uint64_t destinationOffset = tileIndex * (uint64_t)tileSizeBytes;
						for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {
							const uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
							if (byteIndex < tileSizeBytes) {
								outputTiles[destinationOffset + byteIndex] = cache[sourceOffset + byteIndex];
							}
						}
						d_releaseDirectMappedCacheSlot(cacheSlotIndexOut, tileCacheSlotLock, localThreadIndex); 
						continue;
					}

					// Cache-miss step 1: streams data from RAM, decodes the data and writes to device memory output
					// Cache-miss step 2: copy the decoded data to the cache slot.
					uint32_t decodeBitIndex = 0;
					const uint32_t one = 1;
					const uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * (uint64_t)tile];
					const uint32_t* treeBlockPtr = &treePtr[512 * tile];
					// Loading tree into smem.
					const uint32_t numNodes = treeBlockPtr[0];
					const uint32_t numTreeLoadingSteps = (1 + numNodes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					for (int l = 0; l < numTreeLoadingSteps; l++) {
						uint32_t node = l * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (node < 1 + numNodes) {
							s_tree[node] = treeBlockPtr[node];
						}
					}
					// Each block is computed by whole block, so this is inside a branch that is taken by whole block.
					__syncthreads();
					// Decode steps.
					uint32_t chunkCache = 0;
					uint32_t chunkCacheIndex = -1;
					for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							unsigned char leafNodeFound = 0;
							uint32_t currentNodeIndex = 0;
							uint8_t symbol = 0;
							while (!leafNodeFound) {
								const uint32_t chunkColumn = localThreadIndex;
								const uint32_t chunkRow = decodeBitIndex >> 5;
								const uint32_t chunkBit = decodeBitIndex & 31;
								// Aggregated access to the unified mem.
								const uint32_t chunkLoadIndex = chunkColumn + chunkRow * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
								if (chunkCacheIndex != chunkLoadIndex) {
									chunkCache = chunkBlockPtr[chunkLoadIndex];
									chunkCacheIndex = chunkLoadIndex;
								}
								
								const uint32_t bitBeingDecoded = (chunkCache >> chunkBit) & one;
								const uint32_t node = s_tree[1 + currentNodeIndex];
								leafNodeFound = (node >> 8) & 0b11111111;
								const uint16_t childNodeStart = node >> 16;
								symbol = node & 0b11111111;
								currentNodeIndex = bitBeingDecoded ? childNodeStart + 1 : childNodeStart;
								decodeBitIndex++;
							}
							decodeBitIndex--;
							// Copying to the output.
							outputTiles[tileIndex * (uint64_t)tileSizeBytes + byteIndex] = symbol;
							// Updating the cache.
							cache[sourceOffset + byteIndex] = symbol;
						}
					}
					d_releaseDirectMappedCacheSlot(cacheSlotIndexOut, tileCacheSlotLock, localThreadIndex);
				}
			}
			
		}
	}
}