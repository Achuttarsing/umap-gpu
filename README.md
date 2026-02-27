# umap-gpu

UMAP dimensionality reduction with HNSW k-nearest-neighbor search and WebGPU-accelerated SGD optimization, with a transparent CPU fallback.

## What it does

Takes a set of high-dimensional vectors and returns a low-dimensional embedding (default: 2D) suitable for visualization or downstream tasks.

The pipeline runs in three stages:

1. **k-NN** — approximate nearest neighbors via [hnswlib-wasm](https://github.com/yoshoku/hnswlib-wasm) (O(n log n))
2. **Fuzzy simplicial set** — builds a weighted graph from the k-NN graph using smooth distances
3. **SGD** — optimizes the embedding using attraction/repulsion forces:
   - **WebGPU** compute shader when available (Chrome 113+, Edge 113+)
   - **CPU** fallback otherwise — identical output, just slower

## Install

```bash
npm install umap-gpu
```

> Requires a browser or runtime with WebGPU support for GPU acceleration. The CPU fallback works anywhere.

## Usage

```ts
import { fit } from 'umap-gpu';

const vectors = [
  [1.0, 0.0, 0.3],
  [0.9, 0.1, 0.4],
  [0.0, 1.0, 0.8],
  // ...
];

const embedding = await fit(vectors);
// Float32Array of length n * nComponents (default: n * 2)
// embedding[i*2], embedding[i*2 + 1] → 2D coordinates of point i
```

### Options

```ts
const embedding = await fit(vectors, {
  nComponents: 2,      // output dimensions (default: 2)
  nNeighbors:  15,     // k-NN graph degree (default: 15)
  nEpochs:     500,    // SGD iterations (default: 500 for <10k points, 200 otherwise)
  minDist:     0.1,    // minimum distance between points in the embedding (default: 0.1)
  spread:      1.0,    // scale of the embedding (default: 1.0)
  hnsw: {
    M:               16,  // HNSW graph connectivity (default: 16)
    efConstruction: 200,  // build-time search width (default: 200)
    efSearch:        50,  // query-time search width (default: 50)
  },
});
```

### Checking GPU availability

```ts
import { isWebGPUAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  console.log('Will use WebGPU-accelerated SGD');
} else {
  console.log('Will fall back to CPU SGD');
}
```

## Build

```bash
npm run build   # compiles TypeScript to dist/
npm test        # runs the unit test suite (Vitest)
```

## Browser support

| Feature | Requirement |
|---------|-------------|
| WebGPU SGD | Chrome 113+, Edge 113+, Safari 18+ |
| CPU fallback | Any modern browser / Node.js |
| HNSW (WASM) | Any environment with WebAssembly support |

## License

MIT
