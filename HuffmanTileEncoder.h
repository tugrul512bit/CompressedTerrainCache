#ifndef _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#define _INCLUDE_GUARD_HUFFMAN_TILE_ENCODER_
#include<vector>
#include<memory>
#include<algorithm>
#include<queue>
namespace HuffmanTileEncoder {
    // This is used for shape of encoded data.
	constexpr int NUM_CUDA_THREADS_PER_BLOCK = 256;
	template<typename T>
	int computeBlockAlignedSize(int tileWidth, int tileHeight) {
		int size = tileWidth * tileHeight;
		int alignedSize = sizeof(T) * size;
		int parallelChunkSize = sizeof(uint32_t) * NUM_CUDA_THREADS_PER_BLOCK;
		while (alignedSize % parallelChunkSize != 0) {
			alignedSize++;
		}
		return alignedSize;
	}
	// For defining area of tile by two corners (top-left, bottom-right)
	struct Rectangle {
		uint64_t x1, y1, x2, y2;
	};
	template<typename T>
	struct Tile {
		Rectangle area;
		std::vector<unsigned char> sourceData;
		std::vector<unsigned char> encodedData;
		std::vector<uint16_t> encodedTree;
		int index;
		void copyInput(uint64_t terrainWidth, uint64_t terrainHeight, T* terrainPtr) {
			int tileWidth = area.x2 - area.x1;
			int tileHeight = area.y2 - area.y1;
			int alignedSize = computeBlockAlignedSize<T>(tileWidth, tileHeight);
			sourceData.resize(sizeof(T) * tileWidth * tileHeight);
			encodedData.resize(alignedSize, 0);
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
			int tileWidth = area.x2 - area.x1;
			int tileHeight = area.y2 - area.y1;
			int alignedSize = computeBlockAlignedSize<T>(tileWidth, tileHeight);
			int histogram[256];
			for (int i = 0; i < 256; i++) {
				histogram[i] = 0;
			}
			for (unsigned char& c : sourceData) {
				histogram[c]++;
			}
			int virtualPadding = alignedSize - sourceData.size();
			if (virtualPadding > 0) {
				histogram[0] += virtualPadding;
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
				// Each 16bit data is made of two 8bit regions, 1st 8bit is value, 2nd 8bit is leaf-node indicator (1 is leaf, 0 is interior node)
				std::vector<uint16_t> linearize() {
					std::vector<uint16_t> output;
					// BFS linearization.
					std::queue<Node*> outputNodes;
					outputNodes.push(this);
					while (outputNodes.size() > 0) {
						Node* current = outputNodes.front();
						outputNodes.pop();
						output.push_back(current->value | (current->leaf << 8));
						if (current->left != nullptr) { 
							outputNodes.push(current->left.get()); 
						}
						if (current->right != nullptr) { 
							outputNodes.push(current->right.get()); 
						}
					}
					return output;
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
			encodedTree = heap[0]->linearize();
			// Encoding in striped pattern. Each row contains same index bits but in chunks of 32 for efficiency.
			// Finding longest column (num32BitSteps) and computing striped pattern.
			int numBytes = sizeof(T) * (area.x2 - area.x1) * (area.y2 - area.y1);
			int numByteSteps = (numBytes + NUM_CUDA_THREADS_PER_BLOCK - 1) / NUM_CUDA_THREADS_PER_BLOCK;
			int currentBit = 0;
			int totalCodeBitsForThread[NUM_CUDA_THREADS_PER_BLOCK];
			int currentCodeBitsForThread[NUM_CUDA_THREADS_PER_BLOCK];
			for (int thread = 0; thread < NUM_CUDA_THREADS_PER_BLOCK; thread++) {
				totalCodeBitsForThread[thread] = 0;
				currentCodeBitsForThread[thread] = 0;
			}
			for (int i = 0; i < numByteSteps; i++) {
				for (int thread = 0; thread < NUM_CUDA_THREADS_PER_BLOCK; thread++) {
					int index = i * NUM_CUDA_THREADS_PER_BLOCK + thread;
					if (index < numBytes) {
						unsigned char code = codeMapping[sourceData[index]];
						unsigned char codeLength = codeLengthMapping[sourceData[index]];
						totalCodeBitsForThread[thread] += codeLength;
					}
					else if (index < alignedSize) {
						unsigned char code = codeMapping[0];
						unsigned char codeLength = codeLengthMapping[0];
						totalCodeBitsForThread[thread] += codeLength;
					}
				}
			}
			int maxBits = 0;
			for (int thread = 0; thread < NUM_CUDA_THREADS_PER_BLOCK; thread++) {
				if (maxBits < totalCodeBitsForThread[thread]) {
					maxBits = totalCodeBitsForThread[thread];
				}
			}
			int num32BitSteps = (maxBits + 32 - 1) / 32;
			unsigned char one = 1;
			for (int i = 0; i < numByteSteps; i++) {
				for (int thread = 0; thread < NUM_CUDA_THREADS_PER_BLOCK; thread++) {
					int index = i * NUM_CUDA_THREADS_PER_BLOCK + thread;
					if (index < numBytes) {
						unsigned char code = codeMapping[sourceData[index]];
						unsigned char codeLength = codeLengthMapping[sourceData[index]];
						for (int bit = 0; bit < codeLength; bit++) {
							// Inside an integer.
							int bitPos = currentCodeBitsForThread[thread] % 32;
							// Inside a column of integers.
							int row = currentCodeBitsForThread[thread] / 32;
							int col = NUM_CUDA_THREADS_PER_BLOCK;
							
							uint32_t data = reinterpret_cast<uint32_t*>(encodedData.data())[col + row * NUM_CUDA_THREADS_PER_BLOCK];
							data = data | (((code >> bitPos) & one) << bitPos);
							reinterpret_cast<uint32_t*>(encodedData.data())[col + row * NUM_CUDA_THREADS_PER_BLOCK] = data;
							currentCodeBitsForThread[thread]++;
						}
					}	else if (index < alignedSize) {
						unsigned char code = codeMapping[0];
						unsigned char codeLength = codeLengthMapping[0];
						for (int bit = 0; bit < codeLength; bit++) {
							// Inside an integer.
							int bitPos = currentCodeBitsForThread[thread] % 32;
							// Inside a column of integers.
							int row = currentCodeBitsForThread[thread] / 32;
							int col = NUM_CUDA_THREADS_PER_BLOCK;

							uint32_t data = reinterpret_cast<uint32_t*>(encodedData.data())[col + row * NUM_CUDA_THREADS_PER_BLOCK];
							data = data | (((code >> bitPos) & one) << bitPos);
							reinterpret_cast<uint32_t*>(encodedData.data())[col + row * NUM_CUDA_THREADS_PER_BLOCK] = data;
							currentCodeBitsForThread[thread]++;
						}
					}
				}
			}
			if (alignedSize < (NUM_CUDA_THREADS_PER_BLOCK * num32BitSteps * sizeof(uint32_t))) {
				std::cout << "ERROR: block-aligned size is less than current size." << std::endl;
				exit(1);
			}
		}

		void copyOutput(unsigned char* encodedTilesPtr, unsigned char* encodedTreesPtr) {
			int tileWidth = area.x2 - area.x1;
			int tileHeight = area.y2 - area.y1;
			uint64_t alignedSize = computeBlockAlignedSize<T>(tileWidth, tileHeight);
			uint64_t outputTileOffset = index * alignedSize;
			uint64_t outputTreeOffset = index * (uint64_t)512;
			int treeElements = encodedTree.size();
			uint16_t* treePtr = reinterpret_cast<uint16_t*>(encodedTreesPtr);
			treePtr[outputTreeOffset++] = treeElements;
			for (int m = 0; m < treeElements; m++) {
				treePtr[outputTreeOffset++] = encodedTree[m];
			}
			std::copy(encodedData.begin(), encodedData.end(), encodedTilesPtr + outputTileOffset);
		}
	};
}
#endif