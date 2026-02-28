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
 * A built HNSW index that can be queried to find nearest neighbors in the
 * training data for new (unseen) points — used by UMAP.transform().
 */
export interface HNSWSearchableIndex {
  searchKnn(queryVectors: number[][], nNeighbors: number): KNNResult;
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

/**
 * Like computeKNN, but also returns the built HNSW index so it can be reused
 * later to project new points (used by UMAP.transform()).
 */
export async function computeKNNWithIndex(
  vectors: number[][],
  nNeighbors: number,
  opts: HNSWOptions = {}
): Promise<{ knn: KNNResult; index: HNSWSearchableIndex }> {
  const { M = 16, efConstruction = 200, efSearch = 50 } = opts;

  const lib = await loadHnswlib();
  const dim = vectors[0].length;
  const n = vectors.length;

  const hnswIndex = new lib.HierarchicalNSW('l2', dim, '');
  hnswIndex.initIndex(n, M, efConstruction, 200);
  hnswIndex.setEfSearch(Math.max(efSearch, nNeighbors));
  hnswIndex.addItems(vectors, false);

  // Query all training vectors (same as computeKNN)
  const knnIndices: number[][] = [];
  const knnDistances: number[][] = [];

  for (let i = 0; i < n; i++) {
    const result = hnswIndex.searchKnn(vectors[i], nNeighbors + 1, undefined);
    const filtered = result.neighbors
      .map((idx: number, j: number) => ({ idx, dist: result.distances[j] }))
      .filter(({ idx }: { idx: number }) => idx !== i)
      .slice(0, nNeighbors);

    knnIndices.push(filtered.map(({ idx }: { idx: number }) => idx));
    knnDistances.push(filtered.map(({ dist }: { dist: number }) => dist));
  }

  const searchableIndex: HNSWSearchableIndex = {
    searchKnn(queryVectors: number[][], k: number): KNNResult {
      const indices: number[][] = [];
      const distances: number[][] = [];

      for (const vec of queryVectors) {
        const result = hnswIndex.searchKnn(vec, k, undefined);
        const sorted = result.neighbors
          .map((idx: number, j: number) => ({ idx, dist: result.distances[j] }))
          .sort((a: { dist: number }, b: { dist: number }) => a.dist - b.dist)
          .slice(0, k);

        indices.push(sorted.map(({ idx }: { idx: number }) => idx));
        distances.push(sorted.map(({ dist }: { dist: number }) => dist));
      }

      return { indices, distances };
    },
  };

  return { knn: { indices: knnIndices, distances: knnDistances }, index: searchableIndex };
}
