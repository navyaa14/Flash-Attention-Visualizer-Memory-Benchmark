# Flash Attention Visualizer & Memory Benchmark

A pure NumPy implementation of **Flash Attention-style tiled attention** with online softmax, causal masking, numerical equivalence checks, memory estimation, and Matplotlib visualizations.

This project compares standard scaled dot-product attention with a tiled Flash Attention-style algorithm that avoids materializing the full `N × N` attention matrix.

> Inspired by FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al., 2022.

---

## Project Overview

Standard attention computes:

```text
Attention(Q, K, V) = softmax(QKᵀ / √d) V
