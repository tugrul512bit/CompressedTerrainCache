# CompressedTerrainCache

# What is this tool?
I takes a terrain of POD elements, compresses the terrain, and allows cached access from CUDA device without consuming much video memory. It's a compressed terrain cache.

It efficiently streams terrain data and runs only 1 CUDA kernel (no copy) to return the required parts of terrain while the backing-store (terrain storage) doesn't consume the graphics card memory (VRAM). For example, when a player moves through 10 GB terrain, the graphics card uses only 1-2 GB of VRAM allocated. It balances allocation size with performance. Some cards don't have enough VRAM, and some cards have too slow PCIE bandwidth. Both of these disadvantages are balanced with CompressedTerrainCache.

```C++
uint32_t terrain[1000 * 1000];
// Creates tiles (3x3 sized) and tile cache (2x2 = for 4 tiles maximum) from the initial terrain state as a faster read-only source of terrain lookup.
CompressedTerrainCache::TileManager<uint32_t> tileManager(terrain, 1000, 1000, 3, 3, 2, 2);

// devicePointer_d points to VRAM with data of tile 1, tile 2, tile 3 (these are zero-based), each with its own linear-indexing for its terrain elements.
auto devicePointer_d = tileManager.decodeSelectedTiles({1, 2, 3}, &timeDecode, &dataSizeDecode, &throughputDecode)

// use devicePointer_d on any gpgpu algorithm without extra copying.
yourAlgorithmOnGpu(devicePointer_d, result);

```

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

# Is it computationally intensive?
- When fully streaming (non-caching), Huffman Decoding is the bottleneck and its mostly due to decoding bits one by one (which translates to only 50-150 GB/s decode throughput depending on data).
- When partially streaming (50%-90% caching), GPU memory bandwidth is bottleneck.
- With benchmark scenario using OpenCV and other calculations, the gpu rarely computes and this causes driver to reduce frequency of GPU and VRAM.

# How does the cache work?
- Leader thread of each CUDA block acquires(locks) a slot of the cache (index depends on x, y coordinates of tile)
- If data is found on cache slot, it is copied to output (uses video memory for max throughput)
- If data is not found, compressed data is streamed from PCIE without consuming video memory but at lower throughput
- Then data is decoded and copied to the cache for future and to the output
- Finally, slot is released for other blocks
- Other blocks can't enter same slot during computations (decoding, filling cache) but the cache is 2D direct-mapped cache so neighboring blocks on the terrain acquire different slots of cache.
- Eviction is simple: if slot has same tile-index, it is a cache-hit, if different tile-index is found then its a cache-miss
- Since encoding - decoding is used, cache-miss is still faster than streaming raw terrain data.
- Depending on player movement in a game (or sub-matrix selection in a math library), cache serves as a persistent fast but small storage on video memory.
- The faster the player moves, the more streaming is required. When player is stationary, cache supplies 100% of the tiles fetched.
- Todo: a different kernel should check all cache-misses and cache-hits and sort the requests on miss/hit value (misses first)
- - This would overlap more of the decoding with the cache-output copies as a latency hiding upgrade

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

Benchmarking in main.cu uses extra memory, real use would be 1/3 of current total allocation.
