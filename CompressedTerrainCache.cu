#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.
		__global__ void k_decodeTile(
			const unsigned char* encodedTiles,
			const unsigned char* encodedTrees,
			const uint32_t blockAlignedElements,
			const uint32_t tileSizeBytes,
			const unsigned char* originalTileDataForComparison,
			const uint32_t numTilesToTest,
			const uint32_t terrainWidth,
			const uint32_t terrainHeight,
			const uint32_t tileWidth,
			const uint32_t tileHeight) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			bool error = false;
			__shared__ uint32_t s_tree[512];
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				const uint32_t numDecodeSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
				uint32_t decodeBitIndex = 0;
				const uint32_t one = 1;
				const uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * (uint64_t)tile];
				const uint32_t* treeBlockPtr = &treePtr[512 * tile];

				if (tile < numTilesToTest) {
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
								if (!leafNodeFound) {
									if (bitBeingDecoded) {
										currentNodeIndex = childNodeStart + 1;
									}
									else {
										currentNodeIndex = childNodeStart;
									}
								}
								decodeBitIndex++;
							}
							decodeBitIndex--;
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) {
								if (symbol != originalTileDataForComparison[globalX + globalY * terrainWidth]) {
									error = true;
								}
							}
						}
					}
					__syncthreads();
				}
			}

			if (error) {
				printf("\nERROR!! Decoded data - original data mismatch = %u. \n", error);
			}
		}


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
			const uint32_t tileHeight) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			uint32_t dummyVar = 0;
			bool computed = false;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				if (tile < numTilesToTest) {
					computed = true;
					// Access steps.
					const uint32_t numAccessSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					for (uint32_t accessStep = 0; accessStep < numAccessSteps; accessStep++) {
						const uint32_t byteIndex = accessStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) {
								dummyVar += originalTileDataForComparison[globalX + globalY * terrainWidth];
							}
						}
					}
				}
			}

			if (computed && dummyVar == 0) {
				printf("\nERROR! Original terrain should include at least 1 non-null character. \n");
			}
		}


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
			const uint32_t tileHeight) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			__shared__ uint32_t s_tree[512];
			uint32_t dummyVar = 0;
			bool computed = false;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				const uint32_t numDecodeSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
				uint32_t decodeBitIndex = 0;
				const uint32_t one = 1;
				const uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * (uint64_t)tile];
				const uint32_t* treeBlockPtr = &treePtr[512 * tile];

				if (tile < numTilesToTest) {
					computed = true;
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
								if (!leafNodeFound) {
									if (bitBeingDecoded) {
										currentNodeIndex = childNodeStart + 1;
									}
									else {
										currentNodeIndex = childNodeStart;
									}
								}
								decodeBitIndex++;
							}
							decodeBitIndex--;
							dummyVar += symbol;
						}
					}
					__syncthreads();
				}
			}

			if (computed && dummyVar == 0) {
				printf("\nERROR! Decoded data should have at least 1 non-null character. \n");
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
			const uint32_t* tileIndexList) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			__shared__ uint32_t s_tree[512];
			uint32_t error = 0;
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
								if (!leafNodeFound) {
									if (bitBeingDecoded) {
										currentNodeIndex = childNodeStart + 1;
									}
									else {
										currentNodeIndex = childNodeStart;
									}
								}
								decodeBitIndex++;
							}
							decodeBitIndex--;
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) {
								if (symbol != originalTileDataForComparison[globalX + globalY * terrainWidth]) {
									error++;
								}
							}
						}
					}
					__syncthreads();
				}
			}

			if (error > 0) {
				printf("\nERROR! Decoded data - original data mismatch = %u. \n", error);
			}
		}

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
			const uint32_t* tileIndexList) {
			const uint32_t numTilesX = (terrainWidth + tileWidth - 1) / tileWidth;
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			__shared__ uint32_t s_tree[512];
			uint32_t dummyVar = 0;
			bool hasComputed = false;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					hasComputed = true;
					const uint32_t tile = tileIndexList[tileIndex];
					const uint32_t numAccessSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					for (uint32_t accessStep = 0; accessStep < numAccessSteps; accessStep++) {
						const uint32_t byteIndex = accessStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint64_t globalX = tileGlobalX * (uint64_t)tileWidth + tileLocalX;
							const uint64_t globalY = tileGlobalY * (uint64_t)tileHeight + tileLocalY;
							if (globalX < terrainWidth && globalY < terrainHeight) {
								dummyVar += originalTileDataForComparison[globalX + globalY * terrainWidth];
							}
						}
					}
				}
			}

			if (hasComputed && dummyVar == 0) {
				printf("\nERROR! Decoded data should have at least 1 non-null character. \n");
			}
		}

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
			const uint32_t numTilesY = (terrainHeight + tileHeight - 1) / tileHeight;
			const uint32_t localThreadIndex = threadIdx.x;
			const uint32_t numBlocks = gridDim.x;
			const uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			const uint32_t* treePtr = reinterpret_cast<const uint32_t*>(encodedTrees);
			const uint32_t* tilePtr = reinterpret_cast<const uint32_t*>(encodedTiles);
			const uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			__shared__ uint32_t s_tree[512];
			uint32_t dummyVar = 0;
			bool hasComputed = false;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tileIndex = tileStep * numBlocks + blockIdx.x;
				if (tileIndex < numTilesToTest) {
					hasComputed = true;
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
								if (!leafNodeFound) {
									if (bitBeingDecoded) {
										currentNodeIndex = childNodeStart + 1;
									}
									else {
										currentNodeIndex = childNodeStart;
									}
								}
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