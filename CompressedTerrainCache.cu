#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.
		__global__ void k_decodeTile(unsigned char* encodedTiles, unsigned char* encodedTrees, int blockAlignedSize) {
			int localThreadIndex = threadIdx.x;
			int globalThreadIndex = localThreadIndex + blockIdx.x * HuffmanTileEncoder::NUM_CUDA_THREADS_PER_BLOCK;
			int num32BitChunksPerThread = 1;
			int numBitsTotal = 30;
			__shared__ uint16_t s_tree[512];
			if (globalThreadIndex == 0) {

				printf("\ncuda\n");
				printf("%i ", reinterpret_cast<uint16_t*>(encodedTrees)[0]);
				for (int i = 0; i < 512; i++) {

				}
				for (int i = 0; i < numBitsTotal; i++) {

				}
			}
		}
	}
}