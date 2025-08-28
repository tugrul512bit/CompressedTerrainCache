#ifndef _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#define _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#include<vector>
#include<memory>
#include<algorithm>
namespace HuffmanTileEncoder {
    // This is used for shape of encoded data.
	constexpr int NUM_CUDA_THREADS_PER_BLOCK = 256;
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
			encodedData.resize(sizeof(T) * size, 0);
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
				unsigned char leaf;
				std::shared_ptr<Node> left;
				std::shared_ptr<Node> right;
				std::vector<unsigned char> sequence;
				unsigned char code;
				unsigned char codeLength;
				Node(int ct = 0, unsigned char val = 0, unsigned char leafNode = false):count(ct), value(val), leaf(leafNode){
					left = nullptr;
					right = nullptr;
					code = 0;
					codeLength = 0;
				}
				void build(int depth = 0, std::vector<unsigned char> currentSequence = std::vector<unsigned char>()) {
					if (left != nullptr) {
						std::vector<unsigned char> leftSequence = currentSequence;
						leftSequence.push_back(0);
						left->build(depth + 1, leftSequence);
					}
					if (right != nullptr) {
						std::vector<unsigned char> rightSequence = currentSequence;
						rightSequence.push_back(1);
						right->build(depth + 1, rightSequence);
					}
					if (leaf) {
						int sz = currentSequence.size();
						if (sz > 8) {
							std::cout << "ERROR: code sequence longer than 8 bits." << std::endl;
							exit(1);
						}
						else {
							unsigned char one = 1;
							for(unsigned char i = 0; i < sz; i++) {
								unsigned char val = currentSequence[i];
								code = code | (val << i);
							}
							codeLength = sz;
						}
					}
				}
				void map(unsigned char * mapPtr, unsigned char * mapLengthPtr) {
					if (leaf) {
						mapPtr[value] = code;
						mapLengthPtr[value] = codeLength;
					}
					else {
						if (left != nullptr) {
							left->map(mapPtr, mapLengthPtr);
						}
						if (right != nullptr) {
							right->map(mapPtr, mapLengthPtr);
						}
					}
				}
			};
            // Huffman Tree.
			std::vector<std::shared_ptr<Node>> heap;
			for (int i = 0; i < 256; i++) {
				if (histogram[i] > 0) {
					unsigned char thisIsLeafNode = 1;
					std::shared_ptr<Node> node = std::make_shared<Node>(histogram[i], i, thisIsLeafNode);
					heap.push_back(node);
				}
			}
			while (heap.size() > 1) {
				std::sort(heap.begin(), heap.end(), [](std::shared_ptr<Node> node1, std::shared_ptr<Node> node2) {
					return node1->count < node2->count;
					});
				std::shared_ptr<Node> left = heap[0];
				std::shared_ptr<Node> right = heap[1];
				heap.erase(heap.begin(), std::next(heap.begin(), 2));
				unsigned char thisIsNotLeafNode = 0;
				std::shared_ptr<Node> parent = std::make_shared<Node>(left->count + right->count, -1, thisIsNotLeafNode);
				parent->left = left;
				parent->right = right;
				heap.push_back(parent);
			}
			heap[0]->build();
			unsigned char codeMapping[256];
			unsigned char codeLengthMapping[256];
			heap[0]->map(codeMapping, codeLengthMapping);
			// Encoding in striped pattern. Each row contains same index bits but in chunks of 32 for efficiency.
			// Finding longest column (num32BitSteps) and computing striped pattern.
			int numBytes = sizeof(T) * (area.x2 - area.x1) * (area.y2 - area.y1);
			int numByteSteps = (numBytes + NUM_CUDA_THREADS_PER_BLOCK - 1) / NUM_CUDA_THREADS_PER_BLOCK;
			int currentBit = 0;
			for (int i = 0; i < numByteSteps; i++) {
				for (int thread = 0; thread < NUM_CUDA_THREADS_PER_BLOCK; thread++) {
					int index = i * NUM_CUDA_THREADS_PER_BLOCK + thread;
					if (index < numBytes) {
						unsigned char code = codeMapping[sourceData[index]];
						unsigned char codeLength = codeLengthMapping[sourceData[index]];

						currentBit += codeLength;
					}
				}
			}
			std::cout<< "input bytes = " << sourceData.size() << "  output bits = " << currentBit << std::endl;
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