export interface FuzzyGraph {
  rows: Uint32Array;     // source node indices
  cols: Uint32Array;     // target node indices
  vals: Float32Array;    // edge weights
  nVertices: number;
}

/**
 * Compute the fuzzy simplicial set from kNN results.
 * This builds the high-dimensional graph weights using smooth kNN distances
 * (sigmas, rhos) and symmetrizes with the fuzzy set union operation.
 */
export function computeFuzzySimplicialSet(
  knnIndices: number[][],
  knnDistances: number[][],
  nNeighbors: number,
  setOpMixRatio = 1.0
): FuzzyGraph {
  const n = knnIndices.length;
  const { sigmas, rhos } = smoothKnnDist(knnDistances, nNeighbors);

  const rowList: number[] = [];
  const colList: number[] = [];
  const valList: number[] = [];

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < knnIndices[i].length; j++) {
      const d = knnDistances[i][j];
      const val =
        d <= rhos[i]
          ? 1.0
          : Math.exp(-((d - rhos[i]) / sigmas[i]));
      rowList.push(i);
      colList.push(knnIndices[i][j]);
      valList.push(val);
    }
  }

  // Symmetrize: W + W^T - W * W^T (fuzzy union)
  const combined = symmetrize(rowList, colList, valList, n, setOpMixRatio);
  return { ...combined, nVertices: n };
}

/**
 * Compute the fuzzy weight graph between new (query) points and training points.
 * Used by UMAP.transform() to project unseen data into an existing embedding.
 *
 * Unlike computeFuzzySimplicialSet, this produces a bipartite graph
 * (new points → training points) with no symmetrization.
 *
 * @param knnIndices   - For each new point, the indices of its training neighbors
 * @param knnDistances - For each new point, the distances to those neighbors
 * @param nNeighbors   - Number of neighbors used
 * @returns FuzzyGraph where rows are new-point indices, cols are training-point indices
 */
export function computeTransformFuzzyWeights(
  knnIndices: number[][],
  knnDistances: number[][],
  nNeighbors: number
): FuzzyGraph {
  const nNew = knnIndices.length;
  const { sigmas, rhos } = smoothKnnDist(knnDistances, nNeighbors);

  const rowList: number[] = [];
  const colList: number[] = [];
  const valList: number[] = [];

  for (let i = 0; i < nNew; i++) {
    for (let j = 0; j < knnIndices[i].length; j++) {
      const d = knnDistances[i][j];
      const val =
        d <= rhos[i]
          ? 1.0
          : Math.exp(-((d - rhos[i]) / sigmas[i]));
      rowList.push(i);
      colList.push(knnIndices[i][j]);
      valList.push(val);
    }
  }

  return {
    rows: new Uint32Array(rowList),
    cols: new Uint32Array(colList),
    vals: new Float32Array(valList),
    nVertices: nNew,
  };
}

/**
 * Compute smooth kNN distances using binary search for each point's
 * sigma value, matching the target perplexity log2(k).
 */
function smoothKnnDist(
  knnDistances: number[][],
  k: number
): { sigmas: Float32Array; rhos: Float32Array } {
  const SMOOTH_K_TOLERANCE = 1e-5;
  const n = knnDistances.length;
  const sigmas = new Float32Array(n);
  const rhos = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const dists = knnDistances[i];
    rhos[i] = dists.find((d) => d > 0) ?? 0;

    let lo = 0;
    let hi = Infinity;
    let mid = 1.0;
    const target = Math.log2(k);

    for (let iter = 0; iter < 64; iter++) {
      let psum = 0;
      for (let j = 0; j < dists.length; j++) {
        psum += Math.exp(-Math.max(0, dists[j] - rhos[i]) / mid);
      }
      if (Math.abs(psum - target) < SMOOTH_K_TOLERANCE) break;
      if (psum > target) {
        hi = mid;
        mid = (lo + hi) / 2;
      } else {
        lo = mid;
        mid = hi === Infinity ? mid * 2 : (lo + hi) / 2;
      }
    }
    sigmas[i] = mid;
  }
  return { sigmas, rhos };
}

/**
 * Symmetrize the sparse graph using fuzzy set union:
 * W_sym = mixRatio * (P + Q - P*Q) + (1 - mixRatio) * (P*Q)
 *
 * where P = weight of directed edge (i→j) and Q = weight of (j→i).
 */
function symmetrize(
  rows: number[],
  cols: number[],
  vals: number[],
  n: number,
  mixRatio: number
): { rows: Uint32Array; cols: Uint32Array; vals: Float32Array } {
  // Step 1: store each directed edge weight individually.
  // Use numeric composite key (r * n + c) — safe for n up to ~94M.
  const forward = new Map<number, number>();
  for (let i = 0; i < rows.length; i++) {
    forward.set(rows[i] * n + cols[i], vals[i]);
  }

  // Step 2: for every directed edge, compute the true fuzzy union with its
  // transpose and emit the symmetrized weight.
  const outRows: number[] = [];
  const outCols: number[] = [];
  const outVals: number[] = [];

  for (const [key, P] of forward) {
    const r = Math.floor(key / n);
    const c = key % n;
    const Q = forward.get(c * n + r) ?? 0;
    const union = P + Q - P * Q;
    const inter = P * Q;
    outRows.push(r);
    outCols.push(c);
    outVals.push(mixRatio * union + (1 - mixRatio) * inter);
  }

  return {
    rows: new Uint32Array(outRows),
    cols: new Uint32Array(outCols),
    vals: new Float32Array(outVals),
  };
}
