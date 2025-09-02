# CompressedTerrainCache
- CPU divides terrain data into tiles that have contiguous rows without any stride.
- Each core compresses a tile and writes to a compressed-tile array
- CUDA data-streaming kernel implements a 2D-direct-mapped cache for the tile access from gpu, cached in VRAM
- Cache-hit uses data from device memory (VRAM) with extra compression capability of CUDA-compressible memory in L2 cache
- Cache-miss streams Huffman-encoded-data from unified-memory (RAM), then decodes it within a block (striped pattern, 1 column per CUDA thread).
- Only a CUDA-kernel does all work without any cudaMemcpy command for tiles.

Currently implemented features:
- Huffman decoding for tiles independently in gpu, 1 tile per CUDA block.
- 2d direct-mapped tile-caching on device memory during streaming (backing-store is unified memory, cache hit uses device-memory, cache miss takes data from backing-store, decodes, and updates cache)

When actively streaming edge tiles of visible range from unified memory and using 2D caching for interior (automatic cache-miss or hit handling):
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark.png)

When the dataset fully fits inside the cache:
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark_max_potential.png)

Todo:
- Enable CUDA-compressible-memory for the direct-mapped-cache


Dependencies:
- main.cu uses OpenCV-4 for visual output during benchmarks and maintained by VCPKG.
- CUDA C++ compiler with C++17.
