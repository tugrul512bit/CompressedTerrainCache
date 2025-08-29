#include "CompressedTerrainCache.cuh"
namespace CompressedTerrainCache {
	namespace Kernels {
		// Each block decodes a tile concurrently with each block thread decoding its own column in a striped-pattern.
		__global__ void k_decodeTile(unsigned char* encodedTiles, unsigned char* encodedTrees, int blockAlignedSize) {
			int globalThreadIndex = threadIdx.x + blockIdx.x * blockDim.x;
			if (globalThreadIndex == 0) {
				printf("\ncuda\n");
			}
		}
	}
}