#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.

		__global__ void k_decodeTile(
			unsigned char* encodedTiles,
			unsigned char* encodedTrees,
			uint32_t blockAlignedElements,
			uint32_t tileSizeBytes,
			unsigned char* originalTileDataForComparison,
			uint32_t numTilesToTest,
			uint32_t terrainWidth,
			uint32_t terrainHeight,
			uint32_t tileWidth,
			uint32_t tileHeight) {
			uint32_t numTilesX = (terrainWidth + tileWidth + 1) / tileWidth;
			uint32_t numTilesY = (terrainHeight + tileHeight + 1) / tileHeight;
			uint32_t localThreadIndex = threadIdx.x;
			uint32_t numBlocks = gridDim.x;
			uint32_t globalThreadIndex = localThreadIndex + blockIdx.x * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
			uint32_t numGlobalThreads = HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK * numBlocks;
			uint32_t* treePtr = reinterpret_cast<uint32_t*>(encodedTrees);
			uint32_t* tilePtr = reinterpret_cast<uint32_t*>(encodedTiles);
			uint32_t blockAlignedBytes = blockAlignedElements * sizeof(uint32_t);
			// Tile steps.
			uint32_t numTileSteps = (numTilesToTest + numBlocks - 1) / numBlocks;
			for (uint32_t tileStep = 0; tileStep < numTileSteps; tileStep++) {
				uint32_t tile = tileStep * numBlocks + blockIdx.x;
				if (tile < numTilesToTest) {
					// Decode steps.
					uint32_t numDecodeSteps = (blockAlignedBytes + HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK - 1) / HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
					uint32_t decodeBitIndex = 0;
					uint32_t one = 1; 
					uint32_t* chunkBlockPtr = &tilePtr[blockAlignedElements * tile];
					uint32_t* treeBlockPtr = &treePtr[512 * tile]; 
					for (uint32_t decodeStep = 0; decodeStep < numDecodeSteps; decodeStep++) {

						uint32_t byteIndex = decodeStep * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK + localThreadIndex;
						if (byteIndex < blockAlignedBytes) {
							bool leafNodeFound = false;
							uint32_t currentNodeIndex = 0;
							uint8_t symbol = 0; 
							while (!leafNodeFound) {
								uint32_t chunkColumn = localThreadIndex;
								uint32_t chunkRow = decodeBitIndex / 32;
								uint32_t chunkBit = decodeBitIndex % 32;
								uint32_t chunk = chunkBlockPtr[chunkColumn + chunkRow * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK];
								uint32_t bitBeingDecoded = (chunk >> chunkBit) & one;
								uint32_t node = treeBlockPtr[1 + currentNodeIndex];
								uint8_t leafNode = node >> 8;
								uint16_t childNodeStart = node >> 16;
								if (!leafNode) {
									if (bitBeingDecoded) {
										currentNodeIndex = childNodeStart + 1;
									}	else {
										currentNodeIndex = childNodeStart;
									}
								}
								else {
									symbol = node & 0b11111111;
								}
								decodeBitIndex++;
								leafNodeFound = leafNode;
							}
							if (tileStep == 0) printf("%c", symbol);
							if(tileStep == 0)
							{
								uint32_t blockRow = tile / numTilesX;
								uint32_t blockColumn = tile % numTilesX;
								uint32_t chunkRow = decodeBitIndex / 32;
								uint32_t chunkBit = decodeBitIndex % 32;
								uint32_t col = blockColumn * tileWidth + (byteIndex % tileWidth);
								uint32_t row = blockRow * tileHeight + (byteIndex / tileWidth);
								if (byteIndex < tileSizeBytes && col < terrainWidth && row < terrainHeight) {
									if (symbol != originalTileDataForComparison[tileSizeBytes * tile + byteIndex]) {
										printf("\nERROR! Encoded data - original data mismatch = %u %u. \n", col, row);
									}
								}
							}
						}
					}
				}
			}
		}
	}
}