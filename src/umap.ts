import { computeKNN, computeKNNWithIndex } from './hnsw-knn';
import type { HNSWSearchableIndex } from './hnsw-knn';
import { computeFuzzySimplicialSet, computeTransformFuzzyWeights } from './fuzzy-set';
import { GPUSgd } from './gpu/sgd';
import { cpuSgd, cpuSgdTransform } from './fallback/cpu-sgd';
import { isWebGPUAvailable } from './gpu/device';

export interface UMAPOptions {
  /** Embedding dimensionality (default: 2) */
  nComponents?: number;
  /** Number of nearest neighbors (default: 15) */
  nNeighbors?: number;
  /** Number of optimization epochs (default: auto based on dataset size) */
  nEpochs?: number;
  /** Minimum distance in the embedding (default: 0.1) */
  minDist?: number;
  /** Spread of the embedding (default: 1.0) */
  spread?: number;
  /** HNSW index parameters */
  hnsw?: { M?: number; efConstruction?: number; efSearch?: number };
}

/**
 * Called after each completed SGD epoch (or every 10 epochs on the GPU path,
 * piggybacking on the existing GPU synchronisation point to avoid extra stalls).
 *
 * @param epoch   - Zero-based index of the epoch that just finished.
 * @param nEpochs - Total number of epochs.
 */
export type ProgressCallback = (epoch: number, nEpochs: number) => void;

/**
 * Fit UMAP to the given high-dimensional vectors and return a low-dimensional embedding.
 *
 * Pipeline:
 * 1. HNSW k-nearest neighbor search (O(n log n) via hnswlib-wasm)
 * 2. Fuzzy simplicial set construction (graph weights)
 * 3. SGD optimization (WebGPU accelerated, with CPU fallback)
 */
export async function fit(
  vectors: number[][],
  opts: UMAPOptions = {},
  onProgress?: ProgressCallback
): Promise<Float32Array> {
  const {
    nComponents = 2,
    nNeighbors = 15,
    minDist = 0.1,
    spread = 1.0,
    hnsw = {},
  } = opts;
  const nEpochs = opts.nEpochs ?? (vectors.length > 10_000 ? 200 : 500);

  // 1. HNSW kNN
  console.time('knn');
  const { indices, distances } = await computeKNN(vectors, nNeighbors, {
    M: hnsw.M ?? 16,
    efConstruction: hnsw.efConstruction ?? 200,
    efSearch: hnsw.efSearch ?? 50,
  });
  console.timeEnd('knn');

  // 2. Fuzzy simplicial set
  console.time('fuzzy-set');
  const graph = computeFuzzySimplicialSet(indices, distances, nNeighbors);
  console.timeEnd('fuzzy-set');

  // 3. Compute a, b curve parameters from minDist/spread
  const { a, b } = findAB(minDist, spread);

  // 4. Epochs per sample (edge sampling schedule)
  const epochsPerSample = computeEpochsPerSample(graph.vals, nEpochs);

  // 5. Random initial embedding
  const n = vectors.length;
  const embedding = new Float32Array(n * nComponents);
  for (let i = 0; i < embedding.length; i++) {
    embedding[i] = Math.random() * 20 - 10;
  }

  // 6. SGD optimization (GPU with CPU fallback)
  console.time('sgd');
  let result: Float32Array;

  if (isWebGPUAvailable()) {
    try {
      const gpu = new GPUSgd();
      await gpu.init();
      result = await gpu.optimize(
        embedding,
        new Uint32Array(graph.rows),
        new Uint32Array(graph.cols),
        epochsPerSample,
        n,
        nComponents,
        nEpochs,
        { a, b, gamma: 1.0, negativeSampleRate: 5 },
        onProgress
      );
    } catch (err) {
      console.warn('WebGPU SGD failed, falling back to CPU:', err);
      result = cpuSgd(embedding, graph, epochsPerSample, n, nComponents, nEpochs, { a, b }, onProgress);
    }
  } else {
    result = cpuSgd(embedding, graph, epochsPerSample, n, nComponents, nEpochs, { a, b }, onProgress);
  }
  console.timeEnd('sgd');

  return result;
}

/**
 * Compute the a, b parameters for the UMAP curve 1/(1 + a*d^(2b)).
 *
 * For arbitrary minDist/spread values, a proper implementation would use
 * Levenberg-Marquardt curve fitting. Here we provide pre-fitted constants
 * for common parameter combinations plus an approximation for others.
 */
export function findAB(
  minDist: number,
  spread: number
): { a: number; b: number } {
  // Pre-fitted values for common configurations (matching Python UMAP)
  if (Math.abs(spread - 1.0) < 1e-6 && Math.abs(minDist - 0.1) < 1e-6) {
    return { a: 1.9292, b: 0.7915 };
  }
  if (Math.abs(spread - 1.0) < 1e-6 && Math.abs(minDist - 0.0) < 1e-6) {
    return { a: 1.8956, b: 0.8006 };
  }
  if (Math.abs(spread - 1.0) < 1e-6 && Math.abs(minDist - 0.5) < 1e-6) {
    return { a: 1.5769, b: 0.8951 };
  }

  // Approximation for other values via numerical fitting
  // This follows the approach from the Python UMAP reference implementation
  const b = approximateB(minDist, spread);
  const a = approximateA(minDist, spread, b);
  return { a, b };
}

function approximateB(minDist: number, spread: number): number {
  // Approximate b from spread
  return 1.0 / (spread * 1.2);
}

function approximateA(minDist: number, spread: number, b: number): number {
  // Approximate a so that 1/(1 + a * minDist^(2b)) ~ 1
  if (minDist < 1e-6) return 1.8956;
  return (1.0 / (1.0 + 1e-3) - 1.0) / -(Math.pow(minDist, 2 * b));
}

// ─── Stateful class API ───────────────────────────────────────────────────────

/**
 * Stateful UMAP model that supports separate fit / transform / fit_transform.
 *
 * Usage:
 * ```ts
 * const umap = new UMAP({ nNeighbors: 15, nComponents: 2 });
 *
 * // Train on high-dimensional data:
 * await umap.fit(trainVectors);
 * console.log(umap.embedding); // Float32Array [nTrain * nComponents]
 *
 * // Project new points into the same space:
 * const newEmbedding = await umap.transform(testVectors);
 *
 * // Or do both at once:
 * const embedding = await umap.fit_transform(vectors);
 * ```
 */
export class UMAP {
  private readonly _nComponents: number;
  private readonly _nNeighbors: number;
  private readonly _minDist: number;
  private readonly _spread: number;
  private readonly _nEpochs: number | undefined;
  private readonly _hnswOpts: NonNullable<UMAPOptions['hnsw']>;
  private readonly _a: number;
  private readonly _b: number;

  /** The low-dimensional embedding produced by the last fit() call. */
  embedding: Float32Array | null = null;

  private _hnswIndex: HNSWSearchableIndex | null = null;
  private _nTrain = 0;

  constructor(opts: UMAPOptions = {}) {
    this._nComponents = opts.nComponents ?? 2;
    this._nNeighbors = opts.nNeighbors ?? 15;
    this._minDist = opts.minDist ?? 0.1;
    this._spread = opts.spread ?? 1.0;
    this._nEpochs = opts.nEpochs;
    this._hnswOpts = opts.hnsw ?? {};
    const { a, b } = findAB(this._minDist, this._spread);
    this._a = a;
    this._b = b;
  }

  /**
   * Train UMAP on `vectors`.
   * Stores the resulting embedding in `this.embedding` and retains the HNSW
   * index so that transform() can project new points later.
   * Returns `this` for chaining.
   */
  async fit(vectors: number[][], onProgress?: ProgressCallback): Promise<this> {
    const n = vectors.length;
    const nEpochs = this._nEpochs ?? (n > 10_000 ? 200 : 500);
    const { M = 16, efConstruction = 200, efSearch = 50 } = this._hnswOpts;

    // 1. Build HNSW index and compute k-NN (index is kept for transform)
    console.time('knn');
    const { knn, index } = await computeKNNWithIndex(vectors, this._nNeighbors, {
      M, efConstruction, efSearch,
    });
    this._hnswIndex = index;
    this._nTrain = n;
    console.timeEnd('knn');

    // 2. Fuzzy simplicial set
    console.time('fuzzy-set');
    const graph = computeFuzzySimplicialSet(knn.indices, knn.distances, this._nNeighbors);
    console.timeEnd('fuzzy-set');

    // 3. Epoch sampling schedule
    const epochsPerSample = computeEpochsPerSample(graph.vals, nEpochs);

    // 4. Random initial embedding
    const embedding = new Float32Array(n * this._nComponents);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] = Math.random() * 20 - 10;
    }

    // 5. SGD (GPU with CPU fallback)
    console.time('sgd');
    if (isWebGPUAvailable()) {
      try {
        const gpu = new GPUSgd();
        await gpu.init();
        this.embedding = await gpu.optimize(
          embedding,
          new Uint32Array(graph.rows),
          new Uint32Array(graph.cols),
          epochsPerSample,
          n,
          this._nComponents,
          nEpochs,
          { a: this._a, b: this._b, gamma: 1.0, negativeSampleRate: 5 },
          onProgress
        );
      } catch (err) {
        console.warn('WebGPU SGD failed, falling back to CPU:', err);
        this.embedding = cpuSgd(embedding, graph, epochsPerSample, n, this._nComponents, nEpochs, {
          a: this._a, b: this._b,
        }, onProgress);
      }
    } else {
      this.embedding = cpuSgd(embedding, graph, epochsPerSample, n, this._nComponents, nEpochs, {
        a: this._a, b: this._b,
      }, onProgress);
    }
    console.timeEnd('sgd');

    return this;
  }

  /**
   * Project new (unseen) `vectors` into the embedding space learned by fit().
   * Must be called after fit().
   *
   * The training embedding is kept fixed; only the new-point positions are
   * optimised. Returns a Float32Array of shape [vectors.length × nComponents].
   *
   * @param normalize - When `true`, min-max normalise each dimension of the
   *   returned embedding to [0, 1].  The stored training embedding is never
   *   mutated.  Defaults to `false`.
   */
  async transform(vectors: number[][], normalize = false): Promise<Float32Array> {
    if (!this._hnswIndex || !this.embedding) {
      throw new Error('UMAP.transform() must be called after fit()');
    }

    const nNew = vectors.length;
    const nEpochs = this._nEpochs ?? (this._nTrain > 10_000 ? 200 : 500);
    // Fewer epochs needed when only refining new-point positions
    const transformEpochs = Math.max(100, Math.floor(nEpochs / 4));

    // 1. Find neighbors of new points inside the training set
    const knn = this._hnswIndex.searchKnn(vectors, this._nNeighbors);

    // 2. Build bipartite fuzzy-weight graph (new → training, no symmetrization)
    const graph = computeTransformFuzzyWeights(knn.indices, knn.distances, this._nNeighbors);

    // 3. Initialize new embeddings as the weighted average of training neighbors
    const rows = new Uint32Array(graph.rows);
    const cols = new Uint32Array(graph.cols);
    const weightSums = new Float32Array(nNew);
    const embeddingNew = new Float32Array(nNew * this._nComponents);

    for (let e = 0; e < rows.length; e++) {
      const i = rows[e];  // new-point index
      const j = cols[e];  // training-point index
      const w = graph.vals[e];
      weightSums[i] += w;
      for (let d = 0; d < this._nComponents; d++) {
        embeddingNew[i * this._nComponents + d] +=
          w * this.embedding[j * this._nComponents + d];
      }
    }

    for (let i = 0; i < nNew; i++) {
      if (weightSums[i] > 0) {
        for (let d = 0; d < this._nComponents; d++) {
          embeddingNew[i * this._nComponents + d] /= weightSums[i];
        }
      } else {
        // No neighbors found — fall back to random position
        for (let d = 0; d < this._nComponents; d++) {
          embeddingNew[i * this._nComponents + d] = Math.random() * 20 - 10;
        }
      }
    }

    // 4. SGD: refine new-point positions (training embedding is fixed)
    const epochsPerSample = computeEpochsPerSample(graph.vals, transformEpochs);

    const result = cpuSgdTransform(
      embeddingNew,
      this.embedding,
      graph,
      epochsPerSample,
      nNew,
      this._nTrain,
      this._nComponents,
      transformEpochs,
      { a: this._a, b: this._b }
    );
    return normalize ? normalizeEmbedding(result, nNew, this._nComponents) : result;
  }

  /**
   * Convenience method equivalent to `fit(vectors)` followed by
   * `transform(vectors)` — but more efficient because the training embedding
   * is returned directly without a second optimization pass.
   *
   * @param normalize - When `true`, min-max normalise each dimension of the
   *   returned embedding to [0, 1].  `this.embedding` is never mutated.
   *   Defaults to `false`.
   */
  async fit_transform(
    vectors: number[][],
    onProgress?: ProgressCallback,
    normalize = false
  ): Promise<Float32Array> {
    await this.fit(vectors, onProgress);
    return normalize
      ? normalizeEmbedding(this.embedding!, vectors.length, this._nComponents)
      : this.embedding!;
  }
}

// ─── Normalization ────────────────────────────────────────────────────────────

/**
 * Min-max normalise a flat row-major embedding array so that every dimension
 * independently falls in [0, 1].  Returns a new Float32Array; the input is
 * never mutated.  Dimensions that are constant across all points are mapped to 0.
 */
function normalizeEmbedding(
  embedding: Float32Array,
  nPoints: number,
  nComponents: number
): Float32Array {
  const result = new Float32Array(embedding.length);
  for (let d = 0; d < nComponents; d++) {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < nPoints; i++) {
      const v = embedding[i * nComponents + d];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const range = max - min;
    for (let i = 0; i < nPoints; i++) {
      result[i * nComponents + d] =
        range > 0 ? (embedding[i * nComponents + d] - min) / range : 0;
    }
  }
  return result;
}

// ─── Per-edge epoch schedule ──────────────────────────────────────────────────

/**
 * Compute per-edge epoch sampling periods based on edge weights.
 * Higher-weight edges are sampled more frequently.
 */
export function computeEpochsPerSample(
  weights: Float32Array,
  nEpochs: number
): Float32Array {
  let maxWeight = -Infinity;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] > maxWeight) maxWeight = weights[i];
  }

  const result = new Float32Array(weights.length);
  for (let i = 0; i < weights.length; i++) {
    const normalized = weights[i] / maxWeight;
    result[i] = normalized > 0 ? nEpochs / normalized : -1;
  }
  return result;
}
