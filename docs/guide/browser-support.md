# Browser Support

umap-gpu works in any modern JavaScript environment. The WebGPU acceleration layer is automatically enabled when available; otherwise the library falls back to a pure CPU implementation with identical output.

## WebGPU Support

| Browser / Runtime | WebGPU SGD | Notes |
|-------------------|-----------|-------|
| Chrome 113+       | ✅        | Stable since May 2023 |
| Edge 113+         | ✅        | Same Chromium base |
| Safari 18+        | ✅        | macOS Sequoia / iOS 18 |
| Firefox           | ⏳        | Behind a flag, not yet stable |
| Node.js           | ❌        | No GPU path (CPU fallback used) |
| Bun               | ❌        | No GPU path (CPU fallback used) |

## CPU Fallback

| Environment | Supported |
|-------------|-----------|
| Any modern browser | ✅ |
| Node.js 18+ | ✅ |
| Bun 1.0+ | ✅ |
| Deno | ✅ |

The CPU fallback is always available and produces bit-for-bit identical embeddings to the GPU path (given the same random seed).

## WASM (HNSW)

The k-NN graph is built with **hnswlib-wasm**, which requires WebAssembly support. All modern browsers and server runtimes support WASM.

## Checking at Runtime

```ts
import { isWebGPUAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  console.log('WebGPU available — using GPU path');
} else {
  console.log('WebGPU not available — using CPU fallback');
}
```

The library calls this check internally; you only need it if you want to branch your own logic based on GPU availability.
