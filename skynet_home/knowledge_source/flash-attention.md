---
title: "FlashAttention: Fast and Memory-Efficient Exact Attention"
authors: ["Tri Dao et al."]
year: 2022
---

# FlashAttention

## Summary

Transformer models are powerful but slow and memory-hungry on long sequences. The bottleneck is that exact attention is quadratic in sequence length. FlashAttention is an exact attention algorithm that reduces memory accesses and is faster than approximate attention.

## The memory wall

GPUs have a small, fast SRAM (~20 MB per Streaming Multiprocessor) and a large, slow HBM (40 GB on an A100). Attention computation involves reading and writing the N x N attention matrix to and from HBM, which dominates wall-clock time on long sequences.

FlashAttention uses two classic techniques — tiling and recomputation — to compute exact attention while never materializing the N x N matrix in HBM. It loads blocks of Q, K, V from slow HBM to fast SRAM, computes attention for that block, and writes the output back to HBM.

## Online softmax / tiling

To avoid materializing the full softmax denominator, FlashAttention computes it block by block using the "online softmax" trick (Milakov & Gimelshein, 2018): the running maximum and running sum are updated incrementally as each new block of keys is processed, and the output is rescaled accordingly.

## Recomputation

The backward pass needs intermediate softmax statistics to compute gradients. Instead of storing the full N x N matrix, FlashAttention recomputes it from Q, K, V in SRAM during the backward pass. This trades compute for memory: O(N^2) FLOPs but O(N) memory.

## FlashAttention-2

FlashAttention-2 improves on the original by reducing non-matmul FLOPs (matmuls are about 2x faster than non-matmul ops on modern GPUs), better partitioning the work along the sequence dimension, and improving occupancy on the GPU. It achieves about 2x speedup over FlashAttention-1.
