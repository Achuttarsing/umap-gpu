# API Reference

## `fit(vectors, opts?, onProgress?)`

Fit UMAP to the given high-dimensional vectors and return a low-dimensional embedding.

**Signature**

```ts
function fit(
  vectors: number[][],
  opts?: UMAPOptions,
  onProgress?: ProgressCallback
): Promise<Float32Array>
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `number[][]` | Array of high-dimensional input points. All vectors must have the same dimensionality. |
| `opts` | `UMAPOptions` | Optional configuration. See [Configuration](/guide/configuration). |
| `onProgress` | `ProgressCallback` | Optional callback invoked after each optimisation epoch. Receives `(epoch, nEpochs)`. |

**Returns**

A `Float32Array` of length `vectors.length × nComponents`. Point `i` occupies indices `[i*nComponents, (i+1)*nComponents)`.

**Example**

```ts
import { fit } from 'umap-gpu';

const embedding = await fit(vectors, { nNeighbors: 15, minDist: 0.1 }, (epoch, total) => {
  console.log(`Epoch ${epoch}/${total}`);
});
// embedding[i*2]   → x of point i
// embedding[i*2+1] → y of point i
```

**Pipeline**

1. HNSW k-nearest neighbor search (`O(n log n)` via hnswlib-wasm)
2. Fuzzy simplicial set construction (graph weights)
3. SGD optimization (WebGPU-accelerated, with CPU fallback)

---

## `class UMAP`

Stateful UMAP model supporting separate `fit` / `transform` / `fit_transform` calls.

### Constructor

```ts
new UMAP(opts?: UMAPOptions)
```

### Properties

#### `embedding`

```ts
embedding: Float32Array | null
```

The low-dimensional embedding produced by the last `fit()` call. `null` before `fit()` is called.

### Methods

#### `fit(vectors, onProgress?)`

Train UMAP on `vectors`. Stores the resulting embedding in `this.embedding` and retains the HNSW index for subsequent `transform()` calls.

```ts
async fit(vectors: number[][], onProgress?: ProgressCallback): Promise<this>
```

Returns `this` for chaining.

#### `transform(vectors, normalize?)`

Project new (unseen) `vectors` into the embedding space learned by `fit()`. The training embedding is kept fixed; only the new-point positions are optimised.

```ts
async transform(vectors: number[][], normalize?: boolean): Promise<Float32Array>
```

Must be called after `fit()`. Throws if the model has not been fitted.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `number[][]` | Array of high-dimensional input points to project. |
| `normalize` | `boolean` | When `true`, min-max normalises each dimension of the returned embedding to [0, 1]. The stored training embedding is never mutated. Default: `false`. |

Returns a `Float32Array` of shape `[vectors.length × nComponents]`.

#### `fit_transform(vectors, onProgress?, normalize?)`

Convenience method equivalent to calling `fit(vectors)` and returning the training embedding. More efficient than calling `fit()` then `transform()` on the same data.

```ts
async fit_transform(
  vectors: number[][],
  onProgress?: ProgressCallback,
  normalize?: boolean
): Promise<Float32Array>
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `number[][]` | Array of high-dimensional input points. All vectors must have the same dimensionality. |
| `onProgress` | `ProgressCallback` | Optional callback invoked after each optimisation epoch. Receives `(epoch, nEpochs)`. |
| `normalize` | `boolean` | When `true`, min-max normalises each dimension of the returned embedding to [0, 1]. The stored training embedding is never mutated. Default: `false`. |

**Example**

```ts
import { UMAP } from 'umap-gpu';

const umap = new UMAP({ nNeighbors: 15, nComponents: 2 });

// Train with progress reporting
await umap.fit(trainVectors, (epoch, total) => {
  console.log(`${epoch}/${total}`);
});
console.log(umap.embedding); // Float32Array [nTrain × 2]

// Project new points
const newEmbedding = await umap.transform(testVectors);

// Or do both in one call:
const embedding = await umap.fit_transform(allVectors);
```

---

## `isWebGPUAvailable()`

Fast synchronous heuristic: returns `true` if `navigator.gpu` exists in the current runtime.

```ts
function isWebGPUAvailable(): boolean
```

> **Note:** A `true` result does **not** guarantee a WebGPU adapter can be
> acquired — `requestAdapter()` may still return `null` (no compatible GPU,
> or the browser has disabled WebGPU for the page).  Use
> `checkWebGPUAvailable()` for a reliable async check.

**Example**

```ts
import { isWebGPUAvailable } from 'umap-gpu';

if (isWebGPUAvailable()) {
  // navigator.gpu exists — GPU path will be attempted
}
```

---

## `checkWebGPUAvailable()`

Reliably check whether WebGPU is usable by attempting to acquire an adapter.
The result is cached — repeated calls are free.

```ts
function checkWebGPUAvailable(): Promise<boolean>
```

**Example**

```ts
import { checkWebGPUAvailable } from 'umap-gpu';

if (await checkWebGPUAvailable()) {
  console.log('WebGPU confirmed — GPU path active');
} else {
  console.log('Falling back to CPU path');
}
```

---

## Types

### `UMAPOptions`

```ts
interface UMAPOptions {
  /** Embedding dimensionality (default: 2) */
  nComponents?: number;
  /** Number of nearest neighbors (default: 15) */
  nNeighbors?: number;
  /** Number of optimization epochs (default: auto — 500 for <10k points, 200 otherwise) */
  nEpochs?: number;
  /** Minimum distance in the embedding (default: 0.1) */
  minDist?: number;
  /** Spread of the embedding (default: 1.0) */
  spread?: number;
  /** HNSW index parameters */
  hnsw?: {
    M?: number;
    efConstruction?: number;
    efSearch?: number;
  };
}
```

### `ProgressCallback`

```ts
type ProgressCallback = (epoch: number, nEpochs: number) => void;
```

A function called after each optimisation epoch. `epoch` is the zero-based
index of the epoch that just completed; `nEpochs` is the total number of
epochs. Use these to compute a fraction: `epoch / nEpochs`.

```ts
const onProgress: ProgressCallback = (epoch, nEpochs) => {
  setProgress(epoch / nEpochs); // value in [0, 1)
};
```

### `KNNResult`

```ts
interface KNNResult {
  indices: number[][];
  distances: number[][];
}
```

### `FuzzyGraph`

```ts
interface FuzzyGraph {
  rows: Uint32Array;
  cols: Uint32Array;
  vals: Float32Array;
  nVertices: number;
}
```
