# ⚠️ NOT READY YET | WORK IN PROGRESS ⚠️

# umap-gpu

UMAP dimensionality reduction with WebGPU-accelerated SGD and HNSW approximate nearest neighbors.

Embed millions of high-dimensional vectors into 2D in seconds — not minutes.

## Why GPU?

The bottleneck in UMAP is the SGD optimization loop: thousands of epochs, millions of edge updates per epoch. On CPU this is sequential. On GPU, all edges run in parallel across thousands of shader cores — expect a significant speedup on large datasets, scaling with both the number of points and the number of epochs.

The k-NN stage uses [hnswlib-wasm](https://github.com/yoshoku/hnswlib-wasm) (O(n log n)) so it stays fast regardless.
A transparent CPU fallback guarantees identical output everywhere WebGPU isn't available.

## Install

```bash
# npm
npm install umap-gpu

# Bun
bun add umap-gpu

# pnpm
pnpm add umap-gpu
```

## Quick start

```ts
import { fit } from 'umap-gpu';

const vectors = [
  [0.1, 0.4, 0.9, ...],  // high-dimensional points
  [0.2, 0.3, 0.8, ...],
  // ...
];

const embedding = await fit(vectors);
// Float32Array — embedding[i*2], embedding[i*2+1] are the 2D coords of point i
```

## Train once, project many times

Use the `UMAP` class to embed a training set and later project new points into the same space without retraining.

```ts
import { UMAP } from 'umap-gpu';

const umap = new UMAP({ nNeighbors: 15, minDist: 0.1 });

// Train
await umap.fit(trainVectors);
console.log(umap.embedding); // Float32Array [nTrain × 2]

// Project new points (training embedding stays fixed)
const projected = await umap.transform(newVectors);
// Float32Array [nNew × 2]
```

## Options

```ts
const umap = new UMAP({
  nComponents: 2,      // output dimensions          (default: 2)
  nNeighbors:  15,     // k-NN graph degree          (default: 15)
  nEpochs:     500,    // SGD iterations             (default: auto — 500 for <10k points, 200 otherwise)
  minDist:     0.1,    // min distance in embedding  (default: 0.1)
  spread:      1.0,    // scale of the embedding     (default: 1.0)
  hnsw: {
    M:               16,  // graph connectivity        (default: 16)
    efConstruction: 200,  // build-time search width  (default: 200)
    efSearch:        50,  // query-time search width  (default: 50)
  },
});

// Same options work with the functional API
const embedding = await fit(vectors, { nNeighbors: 15, minDist: 0.05 });
```

## Check GPU availability

```ts
import { isWebGPUAvailable } from 'umap-gpu';

console.log(isWebGPUAvailable()); // true → GPU path, false → CPU fallback
```

## Browser support

| Feature | Supported in |
|---------|-------------|
| WebGPU SGD | Chrome 113+, Edge 113+, Safari 18+ |
| CPU fallback | Any modern browser / Node.js / Bun |
| HNSW (WASM) | Any environment with WebAssembly |

## Development

```bash
npm test        # Vitest unit tests
npm run build   # TypeScript → dist/
```

## License

MIT
