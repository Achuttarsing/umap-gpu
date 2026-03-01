/**
 * End-to-end output correctness tests for the UMAP WebGPU pipeline.
 *
 * These tests mirror umap-output.test.ts but exercise GPUSgd.optimize()
 * directly, verifying that the WGSL compute shaders and GPU buffer management
 * produce a topologically meaningful embedding.
 *
 * Tests are skipped automatically when WebGPU is unavailable (e.g. the
 * `webgpu` npm package is not installed, or no Vulkan driver is present).
 * The `setup-webgpu.ts` setupFile installs WebGPU globals and sets the
 * __webGPUAvailable flag before this module is evaluated.
 */

import { describe, it, expect, beforeAll } from 'vitest';

// Mock hnsw-knn so gpu/sgd.ts can be imported without WASM.
import { vi } from 'vitest';
vi.mock('../hnsw-knn', () => ({
  computeKNN: vi.fn(),
  computeKNNWithIndex: vi.fn(),
}));

import { computeFuzzySimplicialSet } from '../fuzzy-set';
import { computeEpochsPerSample } from '../umap';
import { GPUSgd } from '../gpu/sgd';

// ─── WebGPU availability ──────────────────────────────────────────────────────

// Set synchronously by setup-webgpu.ts before this module is evaluated.
const webGPUAvailable: boolean =
  (globalThis as Record<string, unknown>).__webGPUAvailable === true;

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Exact (brute-force) k-NN — no WASM required. */
function bruteForceKNN(
  vectors: number[][],
  k: number
): { indices: number[][]; distances: number[][] } {
  const n = vectors.length;
  const dim = vectors[0].length;
  const outIdx: number[][] = [];
  const outDist: number[][] = [];

  for (let i = 0; i < n; i++) {
    const dists: Array<{ idx: number; d: number }> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let d = 0;
      for (let dd = 0; dd < dim; dd++) {
        const diff = vectors[i][dd] - vectors[j][dd];
        d += diff * diff;
      }
      dists.push({ idx: j, d: Math.sqrt(d) });
    }
    dists.sort((a, b) => a.d - b.d);
    outIdx.push(dists.slice(0, k).map((x) => x.idx));
    outDist.push(dists.slice(0, k).map((x) => x.d));
  }

  return { indices: outIdx, distances: outDist };
}

/**
 * Two deterministic clusters, well-separated in `dim`-dimensional space.
 *
 * Cluster A: points oscillate around the origin  (0, 0, …)
 * Cluster B: points oscillate around (CENTER, CENTER, …)
 *
 * Using sin/cos of the point index gives non-collinear points with no RNG.
 */
function makeTwoClusters(
  nPerCluster = 12,
  dim = 6,
  center = 20
): { vectors: number[][]; labels: number[] } {
  const vectors: number[][] = [];
  const labels: number[] = [];

  for (let i = 0; i < nPerCluster; i++) {
    vectors.push(
      Array.from({ length: dim }, (_, d) => Math.sin(i * 1.1 + d * 0.7) * 0.4)
    );
    labels.push(0);
  }
  for (let i = 0; i < nPerCluster; i++) {
    vectors.push(
      Array.from({ length: dim }, (_, d) => center + Math.sin(i * 1.1 + d * 0.7) * 0.4)
    );
    labels.push(1);
  }

  return { vectors, labels };
}

/** Euclidean distance between two 2-D points stored in a flat Float32Array. */
function dist2D(emb: Float32Array, i: number, j: number): number {
  const dx = emb[i * 2] - emb[j * 2];
  const dy = emb[i * 2 + 1] - emb[j * 2 + 1];
  return Math.sqrt(dx * dx + dy * dy);
}

// ─── Constants ────────────────────────────────────────────────────────────────

const N_PER_CLUSTER = 12;
const DIM = 6;
const N_NEIGHBORS = 5;
const N_EPOCHS = 500;
const UMAP_PARAMS = { a: 1.9292, b: 0.7915, gamma: 1.0, negativeSampleRate: 5 };

// ─── GPU suite ────────────────────────────────────────────────────────────────

describe.skipIf(!webGPUAvailable)('UMAP end-to-end output correctness (WebGPU path)', () => {
  // GPUSgd.optimize() is async, so we build the fixture in beforeAll.
  let embedding: Float32Array;
  let labels: number[];
  const n = N_PER_CLUSTER * 2; // 24

  beforeAll(async () => {
    // Install navigator.gpu only in this worker's context.  Keeping this local
    // to beforeAll (rather than the global setupFile) ensures that other test
    // files (e.g. umap-class.test.ts) continue to use the CPU path and their
    // per-epoch progress-callback assertions remain valid.
    const { create } = await import('webgpu');
    const gpu = create([]);
    if (typeof globalThis.navigator === 'undefined') {
      Object.defineProperty(globalThis, 'navigator', {
        value: { gpu },
        configurable: true,
        writable: true,
      });
    } else {
      (globalThis.navigator as { gpu?: unknown }).gpu = gpu;
    }

    const { vectors, labels: l } = makeTwoClusters(N_PER_CLUSTER, DIM);
    labels = l;

    const { indices, distances } = bruteForceKNN(vectors, N_NEIGHBORS);
    const graph = computeFuzzySimplicialSet(indices, distances, N_NEIGHBORS);
    const epochsPerSample = computeEpochsPerSample(graph.vals, N_EPOCHS);

    // Deterministic initial embedding: cluster A near (-4, 0), cluster B near (+4, 0)
    // with small per-point offsets so points are not all on top of each other.
    const initialEmbedding = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const cx = labels[i] === 0 ? -4.0 : 4.0;
      initialEmbedding[i * 2]     = cx + Math.sin(i * 0.9) * 0.5;
      initialEmbedding[i * 2 + 1] = Math.cos(i * 1.3) * 0.5;
    }

    const gpuSgd = new GPUSgd();
    await gpuSgd.init();
    embedding = await gpuSgd.optimize(
      initialEmbedding,
      new Uint32Array(graph.rows),
      new Uint32Array(graph.cols),
      epochsPerSample,
      n,
      2,          // nComponents
      N_EPOCHS,
      UMAP_PARAMS
    );
  });

  // ── 1. Numerical sanity ──────────────────────────────────────────────────

  it('all embedding values are finite (no NaN or Inf)', () => {
    for (let i = 0; i < embedding.length; i++) {
      expect(isFinite(embedding[i])).toBe(true);
    }
  });

  it('embedding is not collapsed (points have non-trivial spread)', () => {
    let sumX = 0;
    for (let i = 0; i < n; i++) sumX += embedding[i * 2];
    const meanX = sumX / n;

    let varX = 0;
    for (let i = 0; i < n; i++) {
      const d = embedding[i * 2] - meanX;
      varX += d * d;
    }
    varX /= n;

    expect(Math.sqrt(varX)).toBeGreaterThan(0.5);
  });

  // ── 2. Topology preservation ─────────────────────────────────────────────

  it('kNN graph only carries within-cluster edges (precondition)', () => {
    const { indices } = bruteForceKNN(
      makeTwoClusters(N_PER_CLUSTER, DIM).vectors,
      N_NEIGHBORS
    );
    for (let i = 0; i < n; i++) {
      for (const j of indices[i]) {
        expect(labels[j]).toBe(labels[i]);
      }
    }
  });

  it('every point is closer to its own cluster centroid than to the other cluster centroid', () => {
    const centA = [0, 0];
    const centB = [0, 0];
    for (let i = 0; i < n; i++) {
      const target = labels[i] === 0 ? centA : centB;
      target[0] += embedding[i * 2];
      target[1] += embedding[i * 2 + 1];
    }
    centA[0] /= N_PER_CLUSTER;  centA[1] /= N_PER_CLUSTER;
    centB[0] /= N_PER_CLUSTER;  centB[1] /= N_PER_CLUSTER;

    for (let i = 0; i < n; i++) {
      const px = embedding[i * 2], py = embedding[i * 2 + 1];
      const dA = Math.hypot(px - centA[0], py - centA[1]);
      const dB = Math.hypot(px - centB[0], py - centB[1]);

      if (labels[i] === 0) {
        expect(dA).toBeLessThan(dB);
      } else {
        expect(dB).toBeLessThan(dA);
      }
    }
  });

  it('mean intra-cluster distance is much less than mean inter-cluster distance', () => {
    let intraSum = 0, intraCount = 0;
    let interSum = 0, interCount = 0;

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = dist2D(embedding, i, j);
        if (labels[i] === labels[j]) {
          intraSum += d;
          intraCount++;
        } else {
          interSum += d;
          interCount++;
        }
      }
    }

    const meanIntra = intraSum / intraCount;
    const meanInter = interSum / interCount;

    expect(meanInter).toBeGreaterThan(meanIntra * 2);
  });

  it('nearest neighbour in embedding belongs to the same cluster for every point', () => {
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let nnLabel = -1;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const d = dist2D(embedding, i, j);
        if (d < minDist) {
          minDist = d;
          nnLabel = labels[j];
        }
      }
      expect(nnLabel).toBe(labels[i]);
    }
  });

  // ── 3. GPU epoch_of_next_sample initialisation (Bug 4) ───────────────────
  //
  // The GPU shader initialises epoch_of_next_sample = epochsPerSample (not 0),
  // matching the Python reference. Verify with a 2-node, 1-epoch run: if the
  // fix is correct, nothing moves; if regressed the edge fires in epoch 0.

  it('GPU: edges do not fire at epoch 0 — epoch_of_next_sample is deferred', async () => {
    const graph = computeFuzzySimplicialSet([[1], [0]], [[0.5], [0.5]], 1);
    const eps = computeEpochsPerSample(graph.vals, 100); // eps[0] >= 1.0
    const initial = new Float32Array([5, 0, -5, 0]);
    const before = new Float32Array(initial);

    const gpuSgd = new GPUSgd();
    await gpuSgd.init();
    const result = await gpuSgd.optimize(
      initial,
      new Uint32Array(graph.rows),
      new Uint32Array(graph.cols),
      eps,
      2,   // nVertices
      2,   // nComponents
      1,   // nEpochs — only epoch 0; edges must not fire
      UMAP_PARAMS
    );

    // Nothing should have moved: all epoch_of_next_sample >= 1.0 > epoch 0.
    for (let i = 0; i < before.length; i++) {
      expect(result[i]).toBe(before[i]);
    }
  });

  it('GPU: edges fire on the second epoch after initialisation', async () => {
    // Same 2-node graph but run 2 epochs. The edge fires at epoch 1.
    // gamma=0 disables repulsion so only attraction forces apply, making the
    // direction of movement deterministic (nodes must move closer together).
    const graph = computeFuzzySimplicialSet([[1], [0]], [[0.5], [0.5]], 1);
    const eps = computeEpochsPerSample(graph.vals, 100);
    const initial = new Float32Array([5, 0, -5, 0]);
    const before = new Float32Array(initial);

    const gpuSgd = new GPUSgd();
    await gpuSgd.init();
    const result = await gpuSgd.optimize(
      initial,
      new Uint32Array(graph.rows),
      new Uint32Array(graph.cols),
      eps,
      2,   // nVertices
      2,   // nComponents
      2,   // nEpochs — epoch 0 (no fire) + epoch 1 (fires)
      { ...UMAP_PARAMS, gamma: 0 }
    );

    let changed = false;
    for (let i = 0; i < before.length; i++) {
      if (Math.abs(result[i] - before[i]) > 1e-4) { changed = true; break; }
    }
    expect(changed).toBe(true);
  });
});
