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
			uint32_t error = 0;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				if (tile < numTilesToTest) {
					// Decode steps.
					const uint32_t numDecodeSteps = (blockAlignedBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					uint32_t decodeBitIndex = 0;
					const uint32_t one = 1;
					const uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * tile];
					const uint32_t* treeBlockPtr = &treePtr[512 * tile];

					for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							unsigned char leafNodeFound = 0;
							uint32_t currentNodeIndex = 0;
							uint8_t symbol = 0;
							while (!leafNodeFound) {
								const uint32_t chunkColumn = localThreadIndex;
								const uint32_t chunkRow = decodeBitIndex / 32;
								const uint32_t chunkBit = decodeBitIndex % 32;
								const uint32_t chunk = chunkBlockPtr[chunkColumn + chunkRow * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK];
								const uint32_t bitBeingDecoded = (chunk >> chunkBit) & one;
								const uint32_t node = treeBlockPtr[1 + currentNodeIndex];
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
							const uint32_t globalX = tileGlobalX * tileWidth + tileLocalX;
							const uint32_t globalY = tileGlobalY * tileHeight + tileLocalY;
							if (symbol != originalTileDataForComparison[globalX + globalY * terrainWidth]) {
								error++;
							}
						}
					}
				}
			}

			if (error > 0) {
				printf("\nERROR! Decoded data - original data mismatch = %u. \n", error);
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
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				if (tile < numTilesToTest) {
					// Access steps.
					const uint32_t numAccessSteps = (tileSizeBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					for (uint32_t accessStep = 0; accessStep < numAccessSteps; accessStep++) {
						const uint32_t byteIndex = accessStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							const uint32_t tileLocalX = byteIndex % tileWidth;
							const uint32_t tileLocalY = byteIndex / tileWidth;
							const uint32_t tileGlobalX = tile % numTilesX;
							const uint32_t tileGlobalY = tile / numTilesX;
							const uint32_t globalX = tileGlobalX * tileWidth + tileLocalX;
							const uint32_t globalY = tileGlobalY * tileHeight + tileLocalY;
							dummyVar += originalTileDataForComparison[globalX + globalY * terrainWidth];
						}
					}
				}
			}

			if (dummyVar == 0) {
				printf("\nERROR! Benchmark test should include at least 1 non-null character. \n");
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
			uint32_t dummyVar = 0;
			// Tile steps.
			const uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				const uint32_t tile = tileStep * numBlocks + blockIdx.x;
				if (tile < numTilesToTest) {
					// Decode steps.
					const uint32_t numDecodeSteps = (blockAlignedBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					uint32_t decodeBitIndex = 0;
					const uint32_t one = 1;
					const uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * tile];
					const uint32_t* treeBlockPtr = &treePtr[512 * tile];

					for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {
						const uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < tileSizeBytes) {
							unsigned char leafNodeFound = 0;
							uint32_t currentNodeIndex = 0;
							uint8_t symbol = 0;
							while (!leafNodeFound) {
								const uint32_t chunkColumn = localThreadIndex;
								const uint32_t chunkRow = decodeBitIndex / 32;
								const uint32_t chunkBit = decodeBitIndex % 32;
								const uint32_t chunk = chunkBlockPtr[chunkColumn + chunkRow * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK];
								const uint32_t bitBeingDecoded = (chunk >> chunkBit) & one;
								const uint32_t node = treeBlockPtr[1 + currentNodeIndex];
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
							const uint32_t globalX = tileGlobalX * tileWidth + tileLocalX;
							const uint32_t globalY = tileGlobalY * tileHeight + tileLocalY;
							dummyVar += symbol;
						}
					}
				}
			}

			if (dummyVar == 0) {
				printf("\nERROR! Decoded data should have at least 1 non-null character. \n");
			}
		}
	}
}