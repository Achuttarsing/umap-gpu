# Configuration

All options are passed through the `UMAPOptions` interface, shared by both the `fit` function and the `UMAP` class constructor.

```ts
interface UMAPOptions {
  nComponents?: number;
  nNeighbors?: number;
  nEpochs?: number;
  minDist?: number;
  spread?: number;
  hnsw?: {
    M?: number;
    efConstruction?: number;
    efSearch?: number;
  };
  backend?: 'webgpu' | 'webgl' | 'cpu';
}
```

## Core Options

### `nComponents`

- **Type**: `number`
- **Default**: `2`

The number of dimensions in the output embedding. Use `2` for 2D scatter plots, `3` for 3D visualization.

### `nNeighbors`

- **Type**: `number`
- **Default**: `15`

The number of nearest neighbors used when constructing the fuzzy topological representation. Higher values produce a more global view of the data structure; lower values focus on local structure.

Typical range: `5` – `50`.

### `nEpochs`

- **Type**: `number`
- **Default**: `500` for ≤ 10 000 points, `200` otherwise

The number of SGD optimization iterations. More epochs generally give a higher-quality embedding at the cost of extra computation time.

### `minDist`

- **Type**: `number`
- **Default**: `0.1`

The minimum distance between points in the 2D embedding. Smaller values allow points to cluster more tightly; larger values push clusters apart and spread the embedding out more uniformly.

Typical range: `0.0` – `0.99`.

### `spread`

- **Type**: `number`
- **Default**: `1.0`

The effective scale of the embedded points. Works in combination with `minDist` to control how tightly points are packed.

### `backend`

- **Type**: `'webgpu' | 'webgl' | 'cpu'`
- **Default**: auto-detected (WebGPU → WebGL → CPU)

Force a specific SGD backend instead of auto-detecting. Useful for testing, debugging, or when you know your target environment.

- `'webgpu'` — WebGPU compute shaders (fastest, requires WebGPU support)
- `'webgl'` — WebGL 2 fallback (wide browser support, >97% coverage)
- `'cpu'` — Pure JavaScript (always available, works in Node.js/Bun/Deno)

## HNSW Options

Passed as a nested object under the `hnsw` key. These control the performance/accuracy trade-off of the k-NN graph construction stage.

### `hnsw.M`

- **Type**: `number`
- **Default**: `16`

The number of bidirectional links created for every new element in the HNSW graph. Higher values improve recall at the cost of memory and build time.

### `hnsw.efConstruction`

- **Type**: `number`
- **Default**: `200`

The size of the dynamic candidate list used during HNSW graph construction. Higher values build a more accurate index but take longer.

### `hnsw.efSearch`

- **Type**: `number`
- **Default**: `50`

The size of the dynamic candidate list used at query time. Increase this to improve recall at the cost of slower queries.

## Example: Full Configuration

```ts
import { UMAP } from 'umap-gpu';

const umap = new UMAP({
  nComponents:  2,       // 2D output
  nNeighbors:   20,      // wider neighborhood
  nEpochs:      300,     // custom epoch count
  minDist:      0.05,    // tighter clusters
  spread:       1.0,
  backend:      'webgl', // force WebGL backend
  hnsw: {
    M:              16,
    efConstruction: 200,
    efSearch:       50,
  },
});

// Functional API accepts the same options:
import { fit } from 'umap-gpu';
const embedding = await fit(vectors, { nNeighbors: 20, minDist: 0.05 });
```
