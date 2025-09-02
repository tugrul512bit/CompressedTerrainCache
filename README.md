# CompressedTerrainCache
- CPU divides terrain data into tiles that have contiguous rows without any stride.
- Each core compresses a tile and writes to a compressed-tile array
- CUDA data-streaming kernel implements a 2D-direct-mapped cache for the tile access from gpu, cached in VRAM
- Cache-hit uses data from device memory (VRAM) with extra compression capability of CUDA-compressible memory in L2 cache
- Cache-miss streams encoded-data from unified-memory (RAM), then decompresses it within a block.
- Only a CUDA-kernel does all work without any cudaMemcpy command for tiles.

Currently implemented features:
- decoding
- 2d direct-mapped tile-caching on device memory during streaming (backing-store is unified memory, cache hit uses device-memory, cache miss takes data from backing-store, decodes, and updates cache)
![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark.png) 
Todo:
- Enable CUDA-compressible-memory for the direct-mapped-cache


Dependencies:
- main.cu uses OpenCV-4 for visual output during benchmarks and maintained by VCPKG.
- CUDA C++ compiler with C++17.
