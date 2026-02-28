---
layout: home

hero:
  name: "umap-gpu"
  text: "GPU-accelerated UMAP"
  tagline: Embed millions of high-dimensional vectors into 2D in seconds — not minutes.
  actions:
    - theme: brand
      text: Get Started
      link: /guide/getting-started
    - theme: alt
      text: API Reference
      link: /guide/api

features:
  - title: WebGPU Acceleration
    details: The SGD optimization loop runs across thousands of GPU shader cores in parallel. Expect dramatic speedups on large datasets compared to CPU-only UMAP.
  - title: HNSW k-NN
    details: Uses hnswlib-wasm for O(n log n) approximate nearest neighbor search — fast and accurate regardless of dataset size.
  - title: Transparent CPU Fallback
    details: When WebGPU is unavailable, umap-gpu silently falls back to an identical CPU implementation. Same API, same output, everywhere.
---
