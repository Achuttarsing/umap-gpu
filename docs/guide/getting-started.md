# Getting Started

## Installation

::: code-group

```bash [bun]
bun add umap-gpu
```

```bash [npm]
npm install umap-gpu
```

```bash [pnpm]
pnpm add umap-gpu
```

:::

## Quick Start

The simplest way to use umap-gpu is the `fit` functional API. Pass an array of high-dimensional vectors and get back a flat `Float32Array` containing the 2D coordinates.

```ts
import { fit } from 'umap-gpu';

const vectors = [
  [0.1, 0.4, 0.9, /* ... */],
  [0.2, 0.3, 0.8, /* ... */],
  // thousands more...
];

const embedding = await fit(vectors);
// embedding[i*2]   → x coordinate of point i
// embedding[i*2+1] → y coordinate of point i
```

The function automatically selects the best available backend (WebGPU → WebGL → CPU) and silently falls back through the chain.

## Train Once, Project Many Times

Use the `UMAP` class when you need to embed a training set and later project new, unseen points into the same space — without retraining.

```ts
import { UMAP } from 'umap-gpu';

const umap = new UMAP({ nNeighbors: 15, minDist: 0.1 });

// Train on your dataset
await umap.fit(trainVectors);
console.log(umap.embedding); // Float32Array [nTrain × 2]

// Project new points (training embedding stays fixed)
const projected = await umap.transform(newVectors);
// Float32Array [nNew × 2]
```

## Check Backend Availability

```ts
import { isWebGPUAvailable, isWebGLAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  console.log('Running on WebGPU');
} else if (isWebGLAvailable()) {
  console.log('Running on WebGL');
} else {
  console.log('Running on CPU fallback');
}
```

You can also check which backend was actually used after fitting:

```ts
await umap.fit(vectors);
console.log(umap.activeBackend); // 'webgpu', 'webgl', or 'cpu'
```
