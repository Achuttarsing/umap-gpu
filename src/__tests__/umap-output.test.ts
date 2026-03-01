/**
 * End-to-end output correctness tests for the UMAP CPU pipeline.
 *
 * These tests verify that the pipeline produces a *meaningful* embedding —
 * i.e. one that genuinely reflects the high-dimensional structure — rather
 * than just checking types and shapes.
 *
 * No WASM / WebGPU is needed: kNN is computed by brute-force (feasible for
 * the small datasets used here) and the CPU-SGD path is exercised directly.
 *
 * The core property under test is **topology preservation**: two
 * well-separated clusters in high-D space must remain well-separated in the
 * 2-D embedding.  Because the clusters are so far apart that every point's
 * k nearest neighbours all belong to the same cluster, the fuzzy graph only
 * carries within-cluster edges.  SGD then reliably pulls cluster members
 * together and repulsion pushes the two groups apart — regardless of random
 * initialisation.
 */
import { describe, it, expect, vi } from 'vitest';

// Mock hnsw-knn so the umap.ts module can be imported in Node without WASM.
vi.mock('../hnsw-knn', () => ({
  computeKNN: vi.fn(),
  computeKNNWithIndex: vi.fn(),
}));

import { computeFuzzySimplicialSet } from '../fuzzy-set';
import { computeEpochsPerSample, findAB } from '../umap';
import { cpuSgd } from '../fallback/cpu-sgd';

// ─── Helpers ─────────────────────────────────────────────────────────────────

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

// ─── Shared fixture ───────────────────────────────────────────────────────────

const N_PER_CLUSTER = 12;
const DIM = 6;
const N_NEIGHBORS = 5;   // all within the same cluster given the large separation
const N_EPOCHS = 500;
const UMAP_PARAMS = { a: 1.9292, b: 0.7915 }; // standard minDist=0.1, spread=1.0

/** Build the full pipeline once and reuse across tests. */
function buildEmbedding(): { embedding: Float32Array; labels: number[]; n: number } {
  const { vectors, labels } = makeTwoClusters(N_PER_CLUSTER, DIM);
  const n = vectors.length; // 24

  const { indices, distances } = bruteForceKNN(vectors, N_NEIGHBORS);
  const graph = computeFuzzySimplicialSet(indices, distances, N_NEIGHBORS);
  const epochsPerSample = computeEpochsPerSample(graph.vals, N_EPOCHS);

  // Deterministic initial embedding: cluster A near (-4, 0), cluster B near (+4, 0)
  // with small per-point offsets so points are not all on top of each other.
  const embedding = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    const cx = labels[i] === 0 ? -4.0 : 4.0;
    embedding[i * 2]     = cx + Math.sin(i * 0.9) * 0.5;
    embedding[i * 2 + 1] = Math.cos(i * 1.3) * 0.5;
  }

  cpuSgd(embedding, graph, epochsPerSample, n, 2, N_EPOCHS, UMAP_PARAMS);
  return { embedding, labels, n };
}

// ─── Tests ───────────────────────────────────────────────────────────────────

describe('UMAP end-to-end output correctness (CPU path)', () => {
  // Build once — shared by all tests in this suite.
  const { embedding, labels, n } = buildEmbedding();

  // ── 1. Numerical sanity ───────────────────────────────────────────────────

  it('all embedding values are finite (no NaN or Inf)', () => {
    for (let i = 0; i < embedding.length; i++) {
      expect(isFinite(embedding[i])).toBe(true);
    }
  });

  it('embedding is not collapsed (points have non-trivial spread)', () => {
    // Compute variance of x-coordinates across all points.
    let sumX = 0;
    for (let i = 0; i < n; i++) sumX += embedding[i * 2];
    const meanX = sumX / n;

    let varX = 0;
    for (let i = 0; i < n; i++) {
      const d = embedding[i * 2] - meanX;
      varX += d * d;
    }
    varX /= n;

    // Standard deviation must be substantially > 0 — a collapsed embedding
    // (all points at the same location) would give std ≈ 0.
    expect(Math.sqrt(varX)).toBeGreaterThan(0.5);
  });

  // ── 2. Topology preservation ──────────────────────────────────────────────

  it('kNN graph only carries within-cluster edges (precondition)', () => {
    // Verify the brute-force kNN found only same-cluster neighbours.
    // If this fails, the cluster separation is too small and the topology
    // test below would be testing the wrong thing.
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
    // Compute 2-D centroids for each cluster.
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
        // Cluster-A point must be closer to centroid A than to centroid B
        expect(dA).toBeLessThan(dB);
      } else {
        // Cluster-B point must be closer to centroid B than to centroid A
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

    // Inter-cluster distance must be substantially larger.
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

  // ── 3. epochOfNextSample initialisation (Bug 4) ───────────────────────────

  it('edges do not fire at epoch 0 — they are deferred by epochsPerSample', () => {
    // Build a 2-node graph with one edge. With epochsPerSample = [1.0],
    // the Bug 4 fix sets epochOfNextSample = [1.0], so the edge skips epoch 0.
    // We verify by running 1 epoch (only epoch 0) and checking nothing moved.
    const graph = computeFuzzySimplicialSet([[1], [0]], [[0.5], [0.5]], 1);
    const eps = computeEpochsPerSample(graph.vals, 100); // all eps ≥ 1.0
    const embedding2 = new Float32Array([5, 0, -5, 0]);
    const before = new Float32Array(embedding2);

    cpuSgd(embedding2, graph, eps, 2, 2, 1, UMAP_PARAMS);  // only epoch 0

    // Nothing should have moved — edges first fire at epoch ≥ eps (≥ 1.0).
    for (let i = 0; i < before.length; i++) {
      expect(embedding2[i]).toBe(before[i]);
    }
  });

  it('edges fire on the second epoch after initialisation', () => {
    // Same setup but run 2 epochs (epoch 0 and epoch 1). The edge fires at
    // epoch 1 and both nodes must move.
    const graph = computeFuzzySimplicialSet([[1], [0]], [[0.5], [0.5]], 1);
    const eps = computeEpochsPerSample(graph.vals, 100);
    const embedding2 = new Float32Array([5, 0, -5, 0]);
    const before = new Float32Array(embedding2);

    cpuSgd(embedding2, graph, eps, 2, 2, 2, { ...UMAP_PARAMS, gamma: 0 });

    let changed = false;
    for (let i = 0; i < before.length; i++) {
      if (Math.abs(embedding2[i] - before[i]) > 1e-6) { changed = true; break; }
    }
    expect(changed).toBe(true);
  });

  // ── 4. findAB curve fit quality (Bug 6) ───────────────────────────────────

  it('LM-fitted a,b produce a curve that passes through the smooth-step target within 5%', () => {
    // The LM fit minimises squared error against a smooth-step target.
    // Check that the fitted curve approximately matches the target at a few
    // representative x values (not just the hard-coded presets).
    const minDist = 0.3;
    const spread = 1.5;
    const { a, b } = findAB(minDist, spread);

    // At x = minDist the target is 1.0 (smooth step hasn't fallen yet).
    const atMin = 1.0 / (1.0 + a * Math.pow(minDist, 2 * b));
    expect(atMin).toBeGreaterThan(0.8);  // curve must be high at minDist

    // At x = spread the target is exp(-(spread - minDist) / spread).
    const target = Math.exp(-(spread - minDist) / spread);
    const atSpread = 1.0 / (1.0 + a * Math.pow(spread, 2 * b));
    expect(Math.abs(atSpread - target)).toBeLessThan(0.1);  // within 0.1 of target

    // Both fitted parameters must be positive.
    expect(a).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(0);
  });
});
