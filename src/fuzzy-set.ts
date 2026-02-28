export interface FuzzyGraph {
  rows: Float32Array;    // source node indices
  cols: Float32Array;    // target node indices
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
    rows: new Float32Array(rowList),
    cols: new Float32Array(colList),
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
      for (let j = 1; j < dists.length; j++) {
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
 * W_sym = W + W^T - W * W^T
 */
function symmetrize(
  rows: number[],
  cols: number[],
  vals: number[],
  _n: number,
  mixRatio: number
): { rows: Float32Array; cols: Float32Array; vals: Float32Array } {
  // Accumulate forward and transpose entries
  const map = new Map<string, number>();
  const addEntry = (r: number, c: number, v: number) => {
    const key = `${r},${c}`;
    const existing = map.get(key) ?? 0;
    map.set(key, existing + v);
  };

  for (let i = 0; i < rows.length; i++) {
    addEntry(rows[i], cols[i], vals[i]);
    addEntry(cols[i], rows[i], vals[i]);
  }

  const outRows: number[] = [];
  const outCols: number[] = [];
  const outVals: number[] = [];

  for (const [key, sum] of map.entries()) {
    const [r, c] = key.split(',').map(Number);
    // Union: P + Q - P*Q (approximated from accumulated sum)
    outRows.push(r);
    outCols.push(c);
    outVals.push(
      sum > 1
        ? mixRatio * (2 - sum) + (1 - mixRatio) * (sum - 1)
        : sum
    );
  }

  return {
    rows: new Float32Array(outRows),
    cols: new Float32Array(outCols),
    vals: new Float32Array(outVals),
  };
}
