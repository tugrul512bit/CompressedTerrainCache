#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.
		__global__ void k_decodeTile(unsigned char* encodedTiles, unsigned char* encodedTrees, int blockAlignedSize) {
			int localThreadIndex = threadIdx.x;
			int globalThreadIndex = localThreadIndex + blockIdx.x * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
			int num32BitChunksPerThread = 1;
			int numBitsTotal = 30;
			__shared__ uint32_t s_tree[512];
			if (globalThreadIndex == 0) {

				printf("\ncuda\n");

				for (int i = 0; i < 512; i++) {

				}
				uint32_t* treePtr = reinterpret_cast<uint32_t*>(encodedTrees);
				uint32_t* tilePtr = reinterpret_cast<uint32_t*>(encodedTiles);
				int numNodes = treePtr[0];
				printf("num nodes = %i \n", numNodes);
				constexpr int n = 256;
				int currentNodeIndex[n];
				for (int i = 0; i < n; i++) currentNodeIndex[i] = 1;

				for (int i = 0; i < numBitsTotal; i++) {
					for (int thr = 0; thr < n; thr++) {
						int chunkIndex = thr + (i / 32) * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
						uint32_t data = tilePtr[chunkIndex];

						int bitIndex = i % 32;
						unsigned char codeBit = (data >> bitIndex) & 1;
						uint32_t node = treePtr[currentNodeIndex[thr]];
						uint8_t symbol = node & 0b11111111;
						uint8_t leafNode = node >> 8;
						uint16_t childNodeStart = node >> 16;

						if (!leafNode) {
							if (codeBit) {
								currentNodeIndex[thr] = childNodeStart + 2;
							}
							else {
								currentNodeIndex[thr] = childNodeStart + 1;
							}
						}

						if (leafNode) {
							printf("%c", symbol);
							currentNodeIndex[thr] = 1;
						}
					}
				}
			}
		}
	}
}