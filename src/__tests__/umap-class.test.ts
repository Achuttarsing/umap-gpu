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

    it('normalizes output to [0,1] per dimension when normalize=true', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      await umap.fit(makeVectors(10, 4));
      const result = await umap.transform(makeVectors(4, 4), true);
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(4 * 2);
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBeGreaterThanOrEqual(0);
        expect(result[i]).toBeLessThanOrEqual(1);
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

    it('normalizes output to [0,1] per dimension when normalize=true', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      const result = await umap.fit_transform(makeVectors(10, 4), undefined, true);
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(10 * 2);
      for (let i = 0; i < result.length; i++) {
        expect(result[i]).toBeGreaterThanOrEqual(0);
        expect(result[i]).toBeLessThanOrEqual(1);
      }
      // this.embedding must remain the raw un-normalized embedding
      expect(result).not.toBe(umap.embedding);
    });
  });

  describe('progress callback', () => {
    it('fit() calls onProgress with (epoch, nEpochs) for each epoch', async () => {
      const nEpochs = 5;
      const umap = new UMAP({ nNeighbors: 3, nEpochs });
      const calls: Array<[number, number]> = [];

      await umap.fit(makeVectors(10, 4), (epoch, total) => {
        calls.push([epoch, total]);
      });

      expect(calls.length).toBe(nEpochs);
      expect(calls[0]).toEqual([0, nEpochs]);
      expect(calls[nEpochs - 1]).toEqual([nEpochs - 1, nEpochs]);
      // epoch values must be strictly increasing
      for (let i = 1; i < calls.length; i++) {
        expect(calls[i][0]).toBe(calls[i - 1][0] + 1);
      }
    });

    it('fit_transform() forwards onProgress to fit()', async () => {
      const nEpochs = 4;
      const umap = new UMAP({ nNeighbors: 3, nEpochs });
      const calls: number[] = [];

      await umap.fit_transform(makeVectors(8, 4), (epoch) => {
        calls.push(epoch);
      });

      expect(calls.length).toBe(nEpochs);
    });

    it('fit() works normally without a callback', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5 });
      await expect(umap.fit(makeVectors(10, 4))).resolves.toBe(umap);
    });
  });
});
