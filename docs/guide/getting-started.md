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

The function automatically uses WebGPU acceleration when available and silently falls back to CPU otherwise.

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

## Check GPU Availability

```ts
import { isWebGPUAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  console.log('Running on GPU');
} else {
  console.log('Running on CPU fallback');
}
```
