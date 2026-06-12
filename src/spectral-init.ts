import type { FuzzyGraph } from './fuzzy-set';

/**
 * Spectral initialization of the low-dimensional embedding.
 *
 * Mirrors the reference implementation's `_spectral_init`: it estimates the
 * leading eigenvectors of the symmetric-normalized graph adjacency
 *   M[r, c] = w[r,c] / sqrt(deg[r]) / sqrt(deg[c])
 * by power iteration with modified Gram-Schmidt, drops the trivial leading
 * (degree) eigenvector, then rescales each coordinate into [0, 10] with a small
 * amount of jitter.
 *
 * The graph is the (directed) forward edge list with symmetrized weights, the
 * same edges later used by SGD. Degrees are accumulated over edge *rows*.
 *
 * Falls back to a small random Gaussian embedding (matching the reference's
 * `try/except` branch) when the result is degenerate / non-finite — e.g. for
 * tiny or disconnected graphs.
 */
export function spectralInit(
  graph: FuzzyGraph,
  n: number,
  nComponents: number,
  rng: () => number
): Float32Array {
  const k = nComponents + 1;
  const { rows, cols, vals } = graph;
  const nEdges = rows.length;

  // Box-Muller Gaussian sampler driven by the shared uniform RNG.
  const randn = (): number => {
    let u = 0;
    while (u === 0) u = rng();
    const v = rng();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  };

  try {
    // Degrees: scatter-add edge weights by row, then d^{-1/2}.
    const degrees = new Float32Array(n);
    for (let e = 0; e < nEdges; e++) degrees[rows[e]] += vals[e];
    const dInvSqrt = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      dInvSqrt[i] = 1.0 / Math.sqrt(Math.max(degrees[i], 1e-10));
    }

    // Normalized edge weights w_norm[e] = vals[e] * d^{-1/2}[r] * d^{-1/2}[c].
    const wNorm = new Float32Array(nEdges);
    for (let e = 0; e < nEdges; e++) {
      wNorm[e] = vals[e] * dInvSqrt[rows[e]] * dInvSqrt[cols[e]];
    }

    // V holds k column vectors of length n; start from random Gaussian noise.
    const V: Float32Array[] = [];
    const next: Float32Array[] = [];
    for (let c = 0; c < k; c++) {
      const col = new Float32Array(n);
      for (let i = 0; i < n; i++) col[i] = randn();
      V.push(col);
      next.push(new Float32Array(n));
    }

    for (let iter = 0; iter < 100; iter++) {
      // Sparse matvec per column: next[rows[e]] += w_norm[e] * V[cols[e]].
      for (let c = 0; c < k; c++) next[c].fill(0);
      for (let e = 0; e < nEdges; e++) {
        const r = rows[e];
        const cc = cols[e];
        const w = wNorm[e];
        for (let c = 0; c < k; c++) next[c][r] += w * V[c][cc];
      }
      for (let c = 0; c < k; c++) V[c].set(next[c]);

      // Modified Gram-Schmidt: orthonormalize the k columns.
      for (let j = 0; j < k; j++) {
        const vj = V[j];
        for (let p = 0; p < j; p++) {
          const vp = V[p];
          let proj = 0;
          for (let r = 0; r < n; r++) proj += vj[r] * vp[r];
          for (let r = 0; r < n; r++) vj[r] -= proj * vp[r];
        }
        let norm = 0;
        for (let r = 0; r < n; r++) norm += vj[r] * vj[r];
        const inv = 1.0 / Math.sqrt(norm + 1e-10);
        for (let r = 0; r < n; r++) vj[r] *= inv;
      }
    }

    // Embedding = columns 1..k-1 (skip the trivial leading eigenvector),
    // expanded so the largest coordinate magnitude is 10, plus tiny jitter.
    const embedding = new Float32Array(n * nComponents);
    let maxAbs = 0;
    for (let d = 0; d < nComponents; d++) {
      const col = V[d + 1];
      for (let i = 0; i < n; i++) {
        const m = Math.abs(col[i]);
        if (m > maxAbs) maxAbs = m;
      }
    }
    const expansion = maxAbs > 0 ? 10.0 / maxAbs : 1.0;
    for (let d = 0; d < nComponents; d++) {
      const col = V[d + 1];
      for (let i = 0; i < n; i++) {
        embedding[i * nComponents + d] = col[i] * expansion + randn() * 0.0001;
      }
    }

    // Per-dimension min-max rescale to [0, 10].
    for (let d = 0; d < nComponents; d++) {
      let mn = Infinity;
      let mx = -Infinity;
      for (let i = 0; i < n; i++) {
        const value = embedding[i * nComponents + d];
        if (value < mn) mn = value;
        if (value > mx) mx = value;
      }
      const range = mx - mn + 1e-10;
      for (let i = 0; i < n; i++) {
        embedding[i * nComponents + d] =
          (10.0 * (embedding[i * nComponents + d] - mn)) / range;
      }
    }

    for (let i = 0; i < embedding.length; i++) {
      if (!Number.isFinite(embedding[i])) {
        throw new Error('non-finite spectral embedding');
      }
    }
    return embedding;
  } catch {
    // Random fallback: small Gaussian, matching the reference's normal * 0.01.
    const embedding = new Float32Array(n * nComponents);
    for (let i = 0; i < embedding.length; i++) embedding[i] = randn() * 0.01;
    return embedding;
  }
}
