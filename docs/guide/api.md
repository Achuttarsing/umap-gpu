# API Reference

## `fit(vectors, opts?)`

Fit UMAP to the given high-dimensional vectors and return a low-dimensional embedding.

**Signature**

```ts
function fit(vectors: number[][], opts?: UMAPOptions): Promise<Float32Array>
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `vectors` | `number[][]` | Array of high-dimensional input points. All vectors must have the same dimensionality. |
| `opts` | `UMAPOptions` | Optional configuration. See [Configuration](/guide/configuration). |

**Returns**

A `Float32Array` of length `vectors.length × nComponents`. Point `i` occupies indices `[i*nComponents, (i+1)*nComponents)`.

**Example**

```ts
import { fit } from 'umap-gpu';

const embedding = await fit(vectors, { nNeighbors: 15, minDist: 0.1 });
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

#### `fit(vectors)`

Train UMAP on `vectors`. Stores the resulting embedding in `this.embedding` and retains the HNSW index for subsequent `transform()` calls.

```ts
async fit(vectors: number[][]): Promise<this>
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
| `onProgress` | `ProgressCallback` | Optional callback invoked after each optimisation epoch with a `[0, 1]` progress value. |
| `normalize` | `boolean` | When `true`, min-max normalises each dimension of the returned embedding to [0, 1]. The stored training embedding is never mutated. Default: `false`. |

**Example**

```ts
import { UMAP } from 'umap-gpu';

const umap = new UMAP({ nNeighbors: 15, nComponents: 2 });

// Train
await umap.fit(trainVectors);
console.log(umap.embedding); // Float32Array [nTrain × 2]

// Project new points
const newEmbedding = await umap.transform(testVectors);

// Or do both in one call:
const embedding = await umap.fit_transform(allVectors);
```

---

## `isWebGPUAvailable()`

Returns `true` if the current runtime exposes a WebGPU adapter (i.e. `navigator.gpu` exists), `false` otherwise.

```ts
function isWebGPUAvailable(): boolean
```

**Example**

```ts
import { isWebGPUAvailable } from 'umap-gpu';

console.log(isWebGPUAvailable()); // true → GPU path, false → CPU fallback
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
type ProgressCallback = (progress: number) => void;
```

A function called after each optimisation epoch. `progress` is a value in `[0, 1]`.

### `KNNResult`

```ts
interface KNNResult {
  indices: Uint32Array;
  distances: Float32Array;
}
```

### `FuzzyGraph`

```ts
interface FuzzyGraph {
  rows: number[];
  cols: number[];
  vals: Float32Array;
}
```
