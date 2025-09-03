# CompressedTerrainCache

# What is this tool?
It's an efficient terrain-streaming tool that runs only 1 CUDA kernel (no copy) to return the required parts of terrain when the terrain doesn't fit graphics card memory (VRAM). For example, when a player moves through 10 GB terrain, the graphics card uses only 1-2 GB of VRAM allocated. It balances allocation size with performance. Some cards don't have enough VRAM, and some cards have too slow PCIE bandwidth. Both of these disadvantages are balanced with CompressedTerrainCache.

# How does it work?
User adds a query with a list of tile indices to be fetched. Then it runs a kernel and the output with tiles (in the same order of the indices given, with fully linear indexing) is returned (inside device-memory so the user can continue working on GPU without a copy).

- Takes a 2D terrain of POD type elements and linearizes each tile of it for faster access later.
- Encodes each tile independently using CPU cores during initialization.
- When kernel is run, it checks if the required list of tile indices are already cached inside the device memory (that is fast), and serves from there.
- When data is not found in cache, it streams encoded tiles from RAM into VRAM and decodes in there.
- Only the encoded data goes through PCIE, effective PCIE bandwidth increases.
- When data is found inside cache, effective bandwidth increases again.
- Huffman encoding is used for all bytes of each tile (its always byte-granularity regardless of POD type of terrain)
- 2D Direct-mapped cache is implemented to optimize for increased cache-hit ratio for spatial-locality of player movements on the 2D terrain.
- Todo: As a last layer of compression, CUDA-compressible-memory is applied to the output.
- Todo: Uses curve-fitting (of player positions in time) to predict player movement on the 2D terrain and starts prefetching the required future tiles while distributing the streaming latency on multiple frames as a low-latency solution.
- Todo: Dynamic parallelism + variable-sized tiles are employed to optimize for unbalanced workloads (such as some tiles doing more decoding with more threads)
- Todo: Aggregated decoding + vectorized device-memory access per thread.
- Todo: Offload more of calculations to the shared-memory lookup-tables.
- Todo: Auto-select the best block-size that maximizes number of resident threads per SM (to increase occupancy).

When actively streaming edge tiles of visible range from unified memory and using 2D caching for interior tiles:

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
