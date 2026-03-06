import { computeKNNWithIndex } from './hnsw-knn';
import type { HNSWSearchableIndex } from './hnsw-knn';
import { computeFuzzySimplicialSet, computeTransformFuzzyWeights } from './fuzzy-set';
import { cpuSgdTransform } from './fallback/cpu-sgd';
import { selectBackend } from './backend';
import type { SGDBackend, BackendType } from './backend';

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
  /** Enable timing instrumentation via console.time/timeEnd (default: false) */
  debug?: boolean;
  /**
   * Force a specific SGD backend instead of auto-detecting.
   * - `'webgpu'` — WebGPU compute shaders (fastest, requires WebGPU support)
   * - `'webgl'`  — WebGL 2 fallback (wide browser support)
   * - `'cpu'`    — Pure JavaScript (always available)
   *
   * When omitted, the best available backend is selected automatically:
   * WebGPU → WebGL → CPU.
   */
  backend?: BackendType;
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
  return new UMAP(opts).fit_transform(vectors, onProgress);
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

  // For any other minDist/spread combination use Levenberg-Marquardt curve
  // fitting, matching the Python reference (scipy.optimize.curve_fit).
  return findABByFitting(minDist, spread);
}

/**
 * Fit a, b by minimising the squared error between 1/(1 + a*x^(2b)) and the
 * smooth-step target function defined by minDist and spread, using the
 * Levenberg-Marquardt (damped least-squares) algorithm.
 *
 * Matches Python UMAP's find_ab_params() which uses scipy.optimize.curve_fit
 * with method='lm' on 300 sample points over [0, 3*spread].
 */
function findABByFitting(minDist: number, spread: number): { a: number; b: number } {
  // Build target: smooth step at minDist with exponential tail scaled by spread.
  const N = 299;
  const xv: number[] = [];
  const yv: number[] = [];
  for (let i = 0; i < N; i++) {
    const x = ((i + 1) / N) * spread * 3;
    xv.push(x);
    yv.push(x < minDist ? 1.0 : Math.exp(-(x - minDist) / spread));
  }

  let a = 1.0;
  let b = 1.0;
  let lambda = 1e-3;  // LM damping

  for (let iter = 0; iter < 500; iter++) {
    // Accumulate J^T r and J^T J for the 2×2 normal equations.
    let jta = 0, jtb = 0;
    let jtja = 0, jtjb = 0, jtjab = 0;
    let resNorm = 0;

    for (let k = 0; k < N; k++) {
      const x = xv[k];
      const xpow = Math.pow(x, 2 * b);
      const denom = 1.0 + a * xpow;
      const pred = 1.0 / denom;
      const res = pred - yv[k];
      resNorm += res * res;

      const denom2 = denom * denom;
      const ja = -xpow / denom2;
      const jb = x > 0 ? -2.0 * Math.log(x) * a * xpow / denom2 : 0.0;

      jta   += ja * res;
      jtb   += jb * res;
      jtja  += ja * ja;
      jtjb  += jb * jb;
      jtjab += ja * jb;
    }

    // Solve (J^T J + lambda * I) * delta = -J^T r
    const h11 = jtja + lambda;
    const h22 = jtjb + lambda;
    const h12 = jtjab;
    const det = h11 * h22 - h12 * h12;
    if (Math.abs(det) < 1e-20) break;

    const da = -(h22 * jta - h12 * jtb) / det;
    const db = -(h11 * jtb - h12 * jta) / det;

    const na = Math.max(1e-4, a + da);
    const nb = Math.max(1e-4, b + db);

    // Evaluate new residual to decide whether to accept the step.
    let newResNorm = 0;
    for (let k = 0; k < N; k++) {
      const xpow = Math.pow(xv[k], 2 * nb);
      const res = 1.0 / (1.0 + na * xpow) - yv[k];
      newResNorm += res * res;
    }

    if (newResNorm < resNorm) {
      a = na;
      b = nb;
      lambda = Math.max(1e-10, lambda / 10);
    } else {
      lambda = Math.min(1e10, lambda * 10);
    }

    if (Math.abs(da) < 1e-8 && Math.abs(db) < 1e-8) break;
  }

  return { a, b };
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
  private readonly _debug: boolean;
  private readonly _backendType: BackendType | undefined;

  /** The low-dimensional embedding produced by the last fit() call. */
  embedding: Float32Array | null = null;

  /** The backend used by the last fit() call (`null` before fit). */
  activeBackend: BackendType | null = null;

  private _hnswIndex: HNSWSearchableIndex | null = null;
  private _nTrain = 0;

  constructor(opts: UMAPOptions = {}) {
    this._nComponents = opts.nComponents ?? 2;
    this._nNeighbors = opts.nNeighbors ?? 15;
    this._minDist = opts.minDist ?? 0.1;
    this._spread = opts.spread ?? 1.0;
    this._nEpochs = opts.nEpochs;
    this._hnswOpts = opts.hnsw ?? {};
    this._debug = opts.debug ?? false;
    this._backendType = opts.backend;
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
    if (this._debug) console.time('knn');
    const { knn, index } = await computeKNNWithIndex(vectors, this._nNeighbors, {
      M, efConstruction, efSearch,
    });
    this._hnswIndex = index;
    this._nTrain = n;
    if (this._debug) console.timeEnd('knn');

    // 2. Fuzzy simplicial set
    if (this._debug) console.time('fuzzy-set');
    const graph = computeFuzzySimplicialSet(knn.indices, knn.distances, this._nNeighbors);
    if (this._debug) console.timeEnd('fuzzy-set');

    // 3. Epoch sampling schedule
    const epochsPerSample = computeEpochsPerSample(graph.vals, nEpochs);

    // 4. Random initial embedding
    const embedding = new Float32Array(n * this._nComponents);
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] = Math.random() * 20 - 10;
    }

    // 5. SGD (auto-select best backend: WebGPU → WebGL → CPU)
    if (this._debug) console.time('sgd');
    const backend = this._backendType
      ? (await import('./backend')).getBackend(this._backendType)
      : await selectBackend();
    this.activeBackend = backend.type;

    this.embedding = await backend.optimize(
      embedding,
      graph,
      epochsPerSample,
      n,
      this._nComponents,
      nEpochs,
      { a: this._a, b: this._b, gamma: 1.0, negativeSampleRate: 5 },
      onProgress,
    );
    if (this._debug) console.timeEnd('sgd');

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
    result[i] = normalized > 0 ? 1.0 / normalized : -1;
  }
  return result;
}
