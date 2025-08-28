#ifndef _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#define _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#include<vector>
#include <functional>
namespace HuffmanTileEncoder {
	// For defining area of tile by two corners (top-left, bottom-right)
	struct Rectangle {
		uint64_t x1, y1, x2, y2;
	};
	template<typename T>
	struct Tile {
		Rectangle area;
		std::vector<unsigned char> sourceData;
		std::vector<unsigned char> encodedData;
		void copyInput(uint64_t terrainWidth, uint64_t terrainHeight, T* terrainPtr) {
			int size = (area.x2 - area.x1) * (area.y2 - area.y1);
			sourceData.resize(sizeof(T) * size);
			uint64_t xMax = (terrainWidth < area.x2 ? terrainWidth : area.x2);
			uint64_t yMax = (terrainHeight < area.y2 ? terrainHeight : area.y2);
			int localIndex = 0;
			for (uint64_t y = area.y1; y < yMax; y++) {
				for (uint64_t x = area.x1; x < xMax; x++) {
					uint64_t index = y * terrainWidth + x;
					sourceData[localIndex++] = terrainPtr[index];
				}
			}
		}
		void encode() {
			int histogram[256];
			for (int i = 0; i < 256; i++) {
				histogram[i] = 0;
			}
			for (unsigned char& c : sourceData) {
				histogram[c]++;
			}
			std::priority_queue<uint32_t, std::vector<uint32_t>, std::greater<uint32_t>> minHeap;
			for (int i = 0; i < 256; i++) {
				if (histogram[i] > 0) {
					minHeap.push(histogram[i]);
				}
			}
			while (!minHeap.empty()) {
				std::cout << minHeap.top() << std::endl;
				minHeap.pop();
			}
		}
	};
	template<typename T>
	struct HuffmanTileEncoder {
		uint64_t terrainWidth;
		uint64_t terrainHeight;
		T* terrainPtr;
	};
}
#endif