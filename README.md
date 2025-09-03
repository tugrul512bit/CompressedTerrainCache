# CompressedTerrainCache

# What is this tool?
It's an efficient terrain-streaming tool that only runs 1 CUDA kernel to return the required parts of terrain when the terrain doesn't fit graphics card memory (VRAM). 
- Takes a 2D terrain of POD type elements and linearizes each tile of it for faster access later.
- Encodes each tile independently using CPU cores during initialization.
- When kernel is run, it checks if the required list of tile indices are already cached inside the device memory (that is fast), and serves from there.
- When data is not found in cache, it streams encoded tiles from RAM into VRAM and decodes in there.
- Only the encoded data goes through PCIE, effective PCIE bandwidth increases.
- When data is found inside cache, effective bandwidth increases again.
- Huffman encoding is used for all bytes of each tile (its always byte-granularity regardless of POD type of terrain)
- 2D Direct-mapped caching is implemented to optimize for increased cache-hit ratio for spatial-locality of player movements on terrain.
- Todo: As a last layer of compression, CUDA-compressible-memory is applied to the output.
- Todo: Uses curve-fitting (of player positions in time) to predict player movement on the 2D terrain and starts prefetching the required future tiles while distributing the streaming latency on multiple frames as a low-latency solution.
- Todo: Dynamic parallelism + variable-sized tiles are employed to optimize for unbalanced workloads (such as some tiles doing more decoding with more threads)

Currently implemented features:
- Huffman decoding for tiles independently in gpu, 1 tile per CUDA block.
- 2d direct-mapped tile-caching on device memory during streaming (backing-store is unified memory, cache hit uses device-memory, cache miss takes data from backing-store, decodes, and updates cache)

When actively streaming edge tiles of visible range from unified memory and using 2D caching for interior (automatic cache-miss or hit handling):

(1 byte per terrain element, PCIE v5.0 x16 lanes, RTX5070)
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark.png)


(4 bytes per terrain element, PCIE v5.0 x16 lanes, RTX5070)
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/wider-POD-type.png)

When the dataset fully fits inside the cache:

(PCIE v4.0 x4 lanes, RTX4070)
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark_max_potential.png)

![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/Algorithm.drawio.png)



Dependencies:
- main.cu uses OpenCV-4 for visual output during benchmarks and maintained by VCPKG.
- CUDA C++ compiler with C++17.
