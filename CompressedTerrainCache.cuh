#ifndef _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#define _INCLUDE_GUARD_COMPRESSED_TERRAIN_CACHE_
#include <vector>
#include <thread>
#include <memory>
#include <queue>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
namespace CompressedTerrainCache {
	// Generates tiles from raw terrain data.
	struct TileGenerator {

	};

	// Encodes tile with Huffman Encoding.
	struct TileCompressor {

	};

	// Contains commands to be sent from main CPU thread to worker threads.
	struct TileCommand {
		int command;
		int parameters[4];
	};

	struct TileWorker {
		std::queue<TileCommand> commandQueue;
		bool working;
		std::thread worker;
		template<typename Lambda>
		TileWorker(Lambda lambda = []() {}) : commandQueue(), working(true), worker(lambda) {

		}
		~TileWorker() {
			worker.join();
		}
	};
	/* 
	Starts dedicated threads for continuous encoding for many tiles with asynchronous scheduling with gpu tasks.
	Uses CUDA unified memory as staging buffer for uploading randomly positioned fine-grained tiles from host to device from much larger storage (RAM).
	*/
	struct TileManager {
		std::vector<std::shared_ptr<TileWorker>> workers;
		TileManager(int numThreads = std::thread::hardware_concurrency()) {
			for (int i = 0; i < numThreads; i++) {
				std::shared_ptr<TileWorker> worker = std::make_shared<TileWorker>([i]() {
					std::cout << "thread " << i << std::endl;
				});
				workers.push_back(worker);
			}
		}

		~TileManager() {

		}
	};
}
#endif