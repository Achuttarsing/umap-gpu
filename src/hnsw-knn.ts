import { loadHnswlib } from 'hnswlib-wasm';

export interface KNNResult {
  indices: number[][];
  distances: number[][];
}

export interface HNSWOptions {
  M?: number;
  efConstruction?: number;
  efSearch?: number;
}

/**
 * Compute k-nearest neighbors using HNSW (Hierarchical Navigable Small World)
 * via hnswlib-wasm, replacing the O(n^2) brute-force search in umap-js with
 * an O(n log n) approximate nearest neighbor search.
 */
export async function computeKNN(
  vectors: number[][],
  nNeighbors: number,
  opts: HNSWOptions = {}
): Promise<KNNResult> {
  const { M = 16, efConstruction = 200, efSearch = 50 } = opts;

  const lib = await loadHnswlib();
  const dim = vectors[0].length;
  const n = vectors.length;

  // HierarchicalNSW(spaceName, numDimensions, autoSaveFilename)
  const index = new lib.HierarchicalNSW('l2', dim, '');
  // initIndex(maxElements, m, efConstruction, randomSeed)
  index.initIndex(n, M, efConstruction, 200);
  index.setEfSearch(Math.max(efSearch, nNeighbors));

  // Add all vectors (returns auto-generated labels)
  index.addItems(vectors, false);

  // Query each vector for its nNeighbors+1 nearest (includes self)
  const knnIndices: number[][] = [];
  const knnDistances: number[][] = [];

  for (let i = 0; i < n; i++) {
    const result = index.searchKnn(vectors[i], nNeighbors + 1, undefined);
    // Remove self (distance ~ 0)
    const filtered = result.neighbors
      .map((idx: number, j: number) => ({ idx, dist: result.distances[j] }))
      .filter(({ idx }: { idx: number }) => idx !== i)
      .slice(0, nNeighbors);

    knnIndices.push(filtered.map(({ idx }: { idx: number }) => idx));
    knnDistances.push(filtered.map(({ dist }: { dist: number }) => dist));
  }

  return { indices: knnIndices, distances: knnDistances };
}
