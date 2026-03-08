# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Development
npm test              # Run all Vitest tests
npm run build         # Bundle with Vite + emit TypeScript declarations (dist/)
npm run docs:dev      # Local VitePress docs server

# Run a single test file
npx vitest run src/__tests__/umap-output.test.ts

# Publish (runs tests + build automatically via prepublishOnly)
npm publish
```

## Architecture

`umap-gpu` is a TypeScript library implementing UMAP dimensionality reduction. The pipeline has three stages:

1. **k-NN** (`src/hnsw-knn.ts`) — uses `hnswlib-wasm` to build an approximate nearest-neighbor graph in O(n log n). The `UMAP` class retains the HNSW index after `fit()` so that `transform()` can project new points without rebuilding.

2. **Fuzzy simplicial set** (`src/fuzzy-set.ts`) — converts the k-NN graph into a weighted edge list (COO format: `rows`, `cols`, `vals` arrays) representing the fuzzy topological structure.

3. **SGD optimization** — two implementations:
   - **GPU path** (`src/gpu/sgd.ts`, `src/gpu/shaders/`) — `GPUSgd` class submits two WebGPU compute passes per epoch: `sgd.wgsl` accumulates gradients into an `atomic<i32>` forces buffer, then `apply-forces.wgsl` applies them to the embedding. Both passes are in the same encoder to guarantee ordering. The GPU device is cached via `src/gpu/device.ts` to avoid exhausting device limits.
   - **CPU fallback** (`src/fallback/cpu-sgd.ts`) — plain TypeScript implementation, used when `isWebGPUAvailable()` returns false or GPU init fails.

**Entry point**: `src/index.ts` re-exports from `src/umap.ts`. The public API is `fit()` (functional) and `UMAP` (stateful class with `fit()` / `transform()` / `fit_transform()`).

## Testing

- `vitest.config.ts` aliases `hnswlib-wasm` to a stub (`src/__tests__/__stubs__/hnswlib-wasm.ts`) so tests run without WASM.
- `src/__tests__/setup-webgpu.ts` installs WebGPU constants globally and sets `globalThis.__webGPUAvailable`. GPU-specific tests use `describe.skipIf(!globalThis.__webGPUAvailable)`.
- **Important**: the setup file deliberately does NOT install `navigator.gpu` globally — only GPU-specific test files do that in their own `beforeAll`, so non-GPU tests don't accidentally take the GPU code path.

## Build output

`npm run build` runs `vite build` (bundles `src/index.ts` → `dist/index.js`, externalizing `hnswlib-wasm`) then `tsc` (emits `.d.ts` declarations). API Extractor rolls up the declarations into `dist/umap-gpu.d.ts` during docs generation.
