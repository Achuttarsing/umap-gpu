/**
 * Cross-implementation comparison tests for WebGPU vs CPU fallback.
 *
 * Both implementations run on *identical* inputs (same kNN graph, same
 * deterministic initial embedding, same hyperparameters) and their outputs
 * are compared using topology-based metrics that are invariant to rotation,
 * reflection, and translation.
 *
 * Exact numerical agreement is NOT expected: the CPU path uses float64
 * arithmetic with sequential per-edge updates while the GPU path uses
 * float32 arithmetic with parallel workgroup updates.  Instead the tests
 * assert *structural* equivalence:
 *
 *   1. Both embeddings separate the two clusters.
 *   2. The pairwise-distance rank order agrees (Spearman r > 0.8).
 *   3. The 5-NN recall between the two embeddings exceeds 80 %.
 *   4. Both agree on the cluster label of every point's nearest neighbour.
 *
 * Tests are skipped automatically when WebGPU is unavailable.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { vi } from 'vitest';

vi.mock('../hnsw-knn', () => ({
  computeKNN: vi.fn(),
  computeKNNWithIndex: vi.fn(),
}));

import { computeFuzzySimplicialSet } from '../fuzzy-set';
import { computeEpochsPerSample } from '../umap';
import { cpuSgd } from '../fallback/cpu-sgd';
import { GPUSgd } from '../gpu/sgd';

// ─── WebGPU availability ──────────────────────────────────────────────────────

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
 * Using sin/cos of the point index avoids RNG while keeping points non-collinear.
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

/**
 * Flat array of all n*(n-1)/2 pairwise distances in canonical (i < j) order.
 */
function pairwiseDistances(emb: Float32Array, n: number): number[] {
  const dists: number[] = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      dists.push(dist2D(emb, i, j));
    }
  }
  return dists;
}

/**
 * Spearman rank correlation of two equal-length arrays.
 * Returns a value in [-1, 1]; close to +1 means the orderings agree.
 */
function spearmanCorrelation(a: number[], b: number[]): number {
  const n = a.length;
  const ra = fractionalRanks(a);
  const rb = fractionalRanks(b);
  const mean = (n + 1) / 2;
  let num = 0, ssA = 0, ssB = 0;
  for (let i = 0; i < n; i++) {
    const da = ra[i] - mean;
    const db = rb[i] - mean;
    num += da * db;
    ssA += da * da;
    ssB += db * db;
  }
  return ssA === 0 || ssB === 0 ? 0 : num / Math.sqrt(ssA * ssB);
}

/** Assign 1-indexed fractional (average) ranks to handle ties. */
function fractionalRanks(arr: number[]): number[] {
  const sorted = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const r = new Array<number>(arr.length);
  for (let pos = 0; pos < sorted.length; ) {
    let end = pos;
    while (end < sorted.length && sorted[end].v === sorted[pos].v) end++;
    const avgRank = (pos + end + 1) / 2; // 1-indexed average rank for the tie group
    for (let k = pos; k < end; k++) r[sorted[k].i] = avgRank;
    pos = end;
  }
  return r;
}

/**
 * k-NN recall: fraction of point i's top-k neighbors in `embA` that also
 * appear in its top-k neighbors in `embB`, averaged over all points.
 */
function knnRecall(
  embA: Float32Array,
  embB: Float32Array,
  n: number,
  k: number
): number {
  let totalRecall = 0;
  for (let i = 0; i < n; i++) {
    const aDists: Array<{ idx: number; d: number }> = [];
    const bDists: Array<{ idx: number; d: number }> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      aDists.push({ idx: j, d: dist2D(embA, i, j) });
      bDists.push({ idx: j, d: dist2D(embB, i, j) });
    }
    aDists.sort((x, y) => x.d - y.d);
    bDists.sort((x, y) => x.d - y.d);
    const aSet = new Set(aDists.slice(0, k).map((x) => x.idx));
    const bSet = new Set(bDists.slice(0, k).map((x) => x.idx));
    let overlap = 0;
    for (const idx of aSet) if (bSet.has(idx)) overlap++;
    totalRecall += overlap / k;
  }
  return totalRecall / n;
}

// ─── Constants ────────────────────────────────────────────────────────────────

const N_PER_CLUSTER = 12;
const DIM = 6;
const N_NEIGHBORS = 5;
const N_EPOCHS = 500;
// Explicit params used by both paths to guarantee identical hyperparameters.
const UMAP_PARAMS = { a: 1.9292, b: 0.7915, gamma: 1.0, negativeSampleRate: 5 };

// ─── Comparison suite ─────────────────────────────────────────────────────────

describe.skipIf(!webGPUAvailable)('WebGPU vs CPU structural equivalence', () => {
  let cpuEmbedding: Float32Array;
  let gpuEmbedding: Float32Array;
  let labels: number[];
  const n = N_PER_CLUSTER * 2; // 24

  beforeAll(async () => {
    // Install navigator.gpu in this worker's context only — same pattern as
    // umap-output-gpu.test.ts to avoid affecting other test files.
    const { create } = await import('webgpu');
    const gpuInstance = create([]);
    if (typeof globalThis.navigator === 'undefined') {
      Object.defineProperty(globalThis, 'navigator', {
        value: { gpu: gpuInstance },
        configurable: true,
        writable: true,
      });
    } else {
      (globalThis.navigator as { gpu?: unknown }).gpu = gpuInstance;
    }

    const { vectors, labels: l } = makeTwoClusters(N_PER_CLUSTER, DIM);
    labels = l;

    const { indices, distances } = bruteForceKNN(vectors, N_NEIGHBORS);
    const graph = computeFuzzySimplicialSet(indices, distances, N_NEIGHBORS);
    const epochsPerSample = computeEpochsPerSample(graph.vals, N_EPOCHS);

    // Shared deterministic initial embedding — both paths start from the same
    // coordinates so any difference in the final embedding reflects only the
    // arithmetic and update-ordering differences between implementations.
    const sharedInitial = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const cx = labels[i] === 0 ? -4.0 : 4.0;
      sharedInitial[i * 2]     = cx + Math.sin(i * 0.9) * 0.5;
      sharedInitial[i * 2 + 1] = Math.cos(i * 1.3) * 0.5;
    }

    // ── CPU path ─────────────────────────────────────────────────────────────
    // cpuSgd mutates the embedding in-place; give it its own copy.
    cpuEmbedding = sharedInitial.slice();
    cpuSgd(cpuEmbedding, graph, epochsPerSample, n, 2, N_EPOCHS, UMAP_PARAMS);

    // ── GPU path ─────────────────────────────────────────────────────────────
    // gpuSgd.optimize() returns a new Float32Array; pass a fresh copy of the
    // initial embedding so both paths start from the same point.
    const gpuSgd = new GPUSgd();
    await gpuSgd.init();
    gpuEmbedding = await gpuSgd.optimize(
      sharedInitial.slice(),
      new Uint32Array(graph.rows),
      new Uint32Array(graph.cols),
      epochsPerSample,
      n,
      2,       // nComponents
      N_EPOCHS,
      UMAP_PARAMS
    );
  });

  // ── 1. Numerical sanity ───────────────────────────────────────────────────

  it('both embeddings contain only finite values (no NaN or Inf)', () => {
    for (let i = 0; i < n * 2; i++) {
      expect(isFinite(cpuEmbedding[i])).toBe(true);
      expect(isFinite(gpuEmbedding[i])).toBe(true);
    }
  });

  // ── 2. Independent cluster structure ─────────────────────────────────────

  it('both embeddings place every point closer to its own cluster centroid', () => {
    for (const emb of [cpuEmbedding, gpuEmbedding]) {
      const centA = [0, 0];
      const centB = [0, 0];
      for (let i = 0; i < n; i++) {
        const t = labels[i] === 0 ? centA : centB;
        t[0] += emb[i * 2];
        t[1] += emb[i * 2 + 1];
      }
      centA[0] /= N_PER_CLUSTER; centA[1] /= N_PER_CLUSTER;
      centB[0] /= N_PER_CLUSTER; centB[1] /= N_PER_CLUSTER;

      for (let i = 0; i < n; i++) {
        const px = emb[i * 2], py = emb[i * 2 + 1];
        const dA = Math.hypot(px - centA[0], py - centA[1]);
        const dB = Math.hypot(px - centB[0], py - centB[1]);
        if (labels[i] === 0) {
          expect(dA).toBeLessThan(dB);
        } else {
          expect(dB).toBeLessThan(dA);
        }
      }
    }
  });

  // ── 3. Cross-implementation agreement ────────────────────────────────────

  it('pairwise distance rank-order agrees between CPU and GPU (Spearman r > 0.8)', () => {
    const cpuDists = pairwiseDistances(cpuEmbedding, n);
    const gpuDists = pairwiseDistances(gpuEmbedding, n);
    const r = spearmanCorrelation(cpuDists, gpuDists);
    expect(r).toBeGreaterThan(0.8);
  });

  it('5-NN recall between CPU and GPU embeddings exceeds 80 %', () => {
    const recall = knnRecall(cpuEmbedding, gpuEmbedding, n, 5);
    expect(recall).toBeGreaterThan(0.8);
  });

  it('both implementations agree on the cluster label of each point\'s nearest neighbour', () => {
    for (let i = 0; i < n; i++) {
      let cpuMinDist = Infinity, cpuNNLabel = -1;
      let gpuMinDist = Infinity, gpuNNLabel = -1;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dc = dist2D(cpuEmbedding, i, j);
        const dg = dist2D(gpuEmbedding, i, j);
        if (dc < cpuMinDist) { cpuMinDist = dc; cpuNNLabel = labels[j]; }
        if (dg < gpuMinDist) { gpuMinDist = dg; gpuNNLabel = labels[j]; }
      }
      // Both implementations must agree with the ground-truth cluster label
      // and with each other.
      expect(cpuNNLabel).toBe(labels[i]);
      expect(gpuNNLabel).toBe(labels[i]);
      expect(cpuNNLabel).toBe(gpuNNLabel);
    }
  });
});
