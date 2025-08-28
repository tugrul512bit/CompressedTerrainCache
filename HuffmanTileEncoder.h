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
		int index;
		void copyInput(uint64_t terrainWidth, uint64_t terrainHeight, T* terrainPtr) {
			int size = (area.x2 - area.x1) * (area.y2 - area.y1);
			sourceData.resize(sizeof(T) * size);
			encodedData.resize(sizeof(T) * size);
			int localIndex = 0;
			T defaultVal = T();
			for (uint64_t y = area.y1; y < area.y2; y++) {
				for (uint64_t x = area.x1; x < area.x2; x++) {
					uint64_t index = y * terrainWidth + x;
					if (x < terrainWidth && y < terrainHeight) {
						reinterpret_cast<T*>(sourceData.data())[localIndex] = terrainPtr[index];
					}
					else {
						reinterpret_cast<T*>(sourceData.data())[localIndex] = defaultVal;
					}
					localIndex++;
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
			struct Node {
				uint32_t count;
				unsigned char value;
				bool operator() (Node& n1, Node& n2) {
					return n1.count < n2.count;
				}
			};
			std::priority_queue<Node, std::vector<Node>, Node> minHeap;
			for (int i = 0; i < 256; i++) {
				if (histogram[i] > 0) {
					Node node;
					node.count = histogram[i];
					node.value = i;
					minHeap.push(node);
				}
			}
			std::cout << "##:: ";
			while (!minHeap.empty()) {
				std::cout<<" { "<< (int)minHeap.top().value<<" " << minHeap.top().count << "} ";
				minHeap.pop();
			}
			encodedData = sourceData;
			std::cout << std::endl;
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