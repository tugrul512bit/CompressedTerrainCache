# CompressedTerrainCache
- CPU divides terrain data into tiles
- Each core compresses a tile and writes to a compressed-tile array
- CUDA entrance kernel implements a 2D-direct-mapped cache for the tile access from gpu, cached in VRAM
- Cache-hit uses data from device memory (VRAM) with extra compression capability of CUDA-compressible memory in L2 cache
- Cache-miss streams data from unified-memory (RAM) that is the compressed tile, then decompresses it within a block
- CPU and GPU works asynchronously to hide latency and improve overall performance


![Screenshot](https://github.com/tugrul512bit/CompressedTerrainCache/blob/master/benchmark.png)
