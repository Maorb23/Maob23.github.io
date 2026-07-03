# Maob23.gitub.io# CUDA Edu - Triton Matvec Visualizer

An interactive visualizer for understanding a Triton matrix-vector multiplication kernel.

## Features

- One Triton program per matrix row
- Interactive M, N, BLOCK_SIZE, and row controls
- Step-by-step code execution
- Visual masks, offsets, loads, multiply, reduce, and store
- Static site: no backend required

## Run locally

```bash
python3 -m http.server 8765
