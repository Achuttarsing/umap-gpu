import { describe, it, expect, vi, beforeEach } from 'vitest';

// Mock hnswlib-wasm (WASM is unavailable in Node) and the computeKNNWithIndex
// wrapper so we can test the UMAP class with a synthetic k-NN result.
vi.mock('../hnsw-knn', () => {
  const fakeIndex = {
    searchKnn: (queryVectors: number[][], nNeighbors: number) => {
      // Return the first nNeighbors training indices (index 0…k-1) for every
      // query point, with distances proportional to the query-point index.
      const n = queryVectors.length;
      const indices: number[][] = [];
      const distances: number[][] = [];
      for (let i = 0; i < n; i++) {
        indices.push(Array.from({ length: nNeighbors }, (_, j) => j));
        distances.push(Array.from({ length: nNeighbors }, (_, j) => (j + 1) * 0.1));
      }
      return { indices, distances };
    },
  };

  const fakeKNNResult = (n: number, nNeighbors: number) => ({
    indices: Array.from({ length: n }, () =>
      Array.from({ length: nNeighbors }, (_, j) => j)
    ),
    distances: Array.from({ length: n }, () =>
      Array.from({ length: nNeighbors }, (_, j) => (j + 1) * 0.1)
    ),
  });

  return {
    computeKNN: vi.fn(async (vectors: number[][], nNeighbors: number) =>
      fakeKNNResult(vectors.length, nNeighbors)
    ),
    computeKNNWithIndex: vi.fn(async (vectors: number[][], nNeighbors: number) => ({
      knn: fakeKNNResult(vectors.length, nNeighbors),
      index: fakeIndex,
    })),
  };
});

import { UMAP } from '../umap';

function makeVectors(n: number, dim: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: dim }, (_, d) => Math.sin(i * 0.3 + d))
  );
}

describe('UMAP class', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('fit()', () => {
    it('stores embedding after fit', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      const vectors = makeVectors(10, 4);
      await umap.fit(vectors);
      expect(umap.embedding).toBeInstanceOf(Float32Array);
      expect(umap.embedding!.length).toBe(10 * 2); // default nComponents=2
    });

    it('returns this for chaining', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      const result = await umap.fit(makeVectors(6, 3));
      expect(result).toBe(umap);
    });

    it('respects nComponents option', async () => {
      const umap = new UMAP({ nComponents: 3, nNeighbors: 3, nEpochs: 5 });
      await umap.fit(makeVectors(8, 4));
      expect(umap.embedding!.length).toBe(8 * 3);
    });
  });

  describe('transform()', () => {
    it('throws if called before fit', async () => {
      const umap = new UMAP({ nNeighbors: 3 });
      await expect(umap.transform(makeVectors(3, 4))).rejects.toThrow();
    });

    it('returns Float32Array of correct shape', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      await umap.fit(makeVectors(10, 4));

      const newVecs = makeVectors(4, 4);
      const result = await umap.transform(newVecs);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(4 * 2); // 4 points × 2D
    });

    it('does not modify the training embedding', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      await umap.fit(makeVectors(10, 4));

      const trainEmbCopy = new Float32Array(umap.embedding!);
      await umap.transform(makeVectors(3, 4));

      for (let i = 0; i < trainEmbCopy.length; i++) {
        expect(umap.embedding![i]).toBe(trainEmbCopy[i]);
      }
    });
  });

  describe('fit_transform()', () => {
    it('returns the training embedding directly', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      const vectors = makeVectors(10, 4);
      const result = await umap.fit_transform(vectors);

      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10 * 2);
      // fit_transform returns the same object stored in umap.embedding
      expect(result).toBe(umap.embedding);
    });
  });
});
