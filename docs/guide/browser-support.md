# Browser Support

umap-gpu works in any modern JavaScript environment. The library automatically selects the best available compute backend for the SGD optimization step, following the fallback chain: **WebGPU → WebGL → CPU**.

## Backend Fallback Chain

```
WebGPU available? ──yes──▶ Use WebGPU (fastest)
       │ no
       ▼
WebGL 2 available? ──yes──▶ Use WebGL (wide support)
       │ no
       ▼
       CPU (always available)
```

## WebGPU Support

| Browser / Runtime | WebGPU SGD | Notes |
|-------------------|-----------|-------|
| Chrome 113+       | ✅        | Stable since May 2023 |
| Edge 113+         | ✅        | Same Chromium base |
| Safari 18+        | ✅        | macOS Sequoia / iOS 18 |
| Firefox           | ⏳        | Behind a flag, not yet stable |
| Node.js           | ❌        | No GPU path (falls back to WebGL or CPU) |
| Bun               | ❌        | No GPU path (falls back to WebGL or CPU) |

## WebGL 2 Support

| Browser / Runtime | WebGL SGD | Notes |
|-------------------|-----------|-------|
| Chrome 56+        | ✅        | Since 2017 |
| Edge 79+          | ✅        | Chromium-based |
| Safari 15+        | ✅        | macOS Monterey / iOS 15 |
| Firefox 51+       | ✅        | Since 2017 |
| Node.js           | ❌        | No canvas/GL (CPU fallback used) |
| Bun               | ❌        | No canvas/GL (CPU fallback used) |

WebGL 2 is supported by **>97%** of browsers in active use, making it an excellent middle-ground fallback when WebGPU is unavailable.

## CPU Fallback

| Environment | Supported |
|-------------|-----------|
| Any modern browser | ✅ |
| Node.js 18+ | ✅ |
| Bun 1.0+ | ✅ |
| Deno | ✅ |

The CPU fallback is always available and produces identical embeddings to the GPU paths (same algorithm, same convergence properties).

## WASM (HNSW)

The k-NN graph is built with **hnswlib-wasm**, which requires WebAssembly support. All modern browsers and server runtimes support WASM.

## Checking at Runtime

```ts
import { isWebGPUAvailable, isWebGLAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  console.log('WebGPU available — using GPU compute shaders');
} else if (isWebGLAvailable()) {
  console.log('WebGL 2 available — using WebGL fallback');
} else {
  console.log('No GPU support — using CPU fallback');
}
```

The library calls these checks internally; you only need them if you want to branch your own logic based on GPU availability.

## Forcing a Specific Backend

You can bypass auto-detection and force a specific backend:

```ts
import { UMAP } from 'umap-gpu';

// Force CPU backend (useful for testing or debugging)
const umap = new UMAP({ backend: 'cpu' });

// Force WebGL backend
const umap2 = new UMAP({ backend: 'webgl' });
```

After calling `fit()`, you can check which backend was used:

```ts
await umap.fit(vectors);
console.log(umap.activeBackend); // 'webgpu', 'webgl', or 'cpu'
```
