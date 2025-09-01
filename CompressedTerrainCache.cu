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
			const uint32_t* tileIndexList,
			unsigned char* outputTiles) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			__shared__ uint32_t s_tree[512];
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					const uint32_t tile = tileIndexList[tileIndex];
					const uint32_t numDecodeSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
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
							outputTiles[tileIndex * (uint64_t)tileSizeBytes + byteIndex] = symbol;
						}
					}
					__syncthreads();
				}
			}
		}
	}
}