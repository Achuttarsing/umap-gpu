/**
 * Tests for the centralized backend API.
 *
 * Verifies that the backend factory, interface, and explicit backend selection
 * work correctly. In the Node.js test environment (no WebGPU/WebGL), the
 * factory should fall through to the CPU backend.
 */

import { describe, it, expect, vi } from 'vitest';

vi.mock('../hnsw-knn', () => {
  const fakeIndex = {
    searchKnn: (queryVectors: number[][], nNeighbors: number) => {
      const n = queryVectors.length;
      return {
        indices: Array.from({ length: n }, () =>
          Array.from({ length: nNeighbors }, (_, j) => j)),
        distances: Array.from({ length: n }, () =>
          Array.from({ length: nNeighbors }, (_, j) => (j + 1) * 0.1)),
      };
    },
  };

  const fakeKNNResult = (n: number, nNeighbors: number) => ({
    indices: Array.from({ length: n }, () =>
      Array.from({ length: nNeighbors }, (_, j) => j)),
    distances: Array.from({ length: n }, () =>
      Array.from({ length: nNeighbors }, (_, j) => (j + 1) * 0.1)),
  });

  return {
    computeKNN: vi.fn(async (vectors: number[][], nNeighbors: number) =>
      fakeKNNResult(vectors.length, nNeighbors)),
    computeKNNWithIndex: vi.fn(async (vectors: number[][], nNeighbors: number) => ({
      knn: fakeKNNResult(vectors.length, nNeighbors),
      index: fakeIndex,
    })),
  };
});

import { selectBackend, getBackend } from '../backend';
import type { SGDBackend, BackendType } from '../backend';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

function makeSimpleGraph(): FuzzyGraph {
  return {
    rows: new Uint32Array([0, 1]),
    cols: new Uint32Array([1, 0]),
    vals: new Float32Array([1.0, 1.0]),
    nVertices: 2,
  };
}

const UMAP_PARAMS = { a: 1.9292, b: 0.7915, gamma: 1.0, negativeSampleRate: 5 };

describe('Backend API', () => {
  describe('selectBackend()', () => {
    it('returns a backend with a valid type', async () => {
      const backend = await selectBackend();
      expect(['webgpu', 'webgl', 'cpu']).toContain(backend.type);
    });

    it('returned backend can optimize an embedding', async () => {
      const backend = await selectBackend();
      const graph = makeSimpleGraph();
      const eps = computeEps(graph.vals, 5);
      const embedding = new Float32Array([5, 0, -5, 0]);

      const result = await backend.optimize(embedding, graph, eps, 2, 2, 5, UMAP_PARAMS);
      expect(result).toBeInstanceOf(Float32Array);
      expect(result.length).toBe(4);
    });
  });

  describe('getBackend()', () => {
    it('returns a CPU backend when requested', () => {
      const backend = getBackend('cpu');
      expect(backend.type).toBe('cpu');
    });

    it('CPU backend produces valid output', async () => {
      const backend = getBackend('cpu');
      const graph = makeSimpleGraph();
      const eps = computeEps(graph.vals, 10);
      const embedding = new Float32Array([5, 0, -5, 0]);
      const initial = new Float32Array(embedding);

      const result = await backend.optimize(embedding, graph, eps, 2, 2, 10, UMAP_PARAMS);
      expect(result).toBeInstanceOf(Float32Array);

      // Should have moved
      let changed = false;
      for (let i = 0; i < result.length; i++) {
        if (Math.abs(result[i] - initial[i]) > 1e-6) { changed = true; break; }
      }
      expect(changed).toBe(true);
    });

    it('all backend types can be instantiated', () => {
      const types: BackendType[] = ['webgpu', 'webgl', 'cpu'];
      for (const t of types) {
        const backend = getBackend(t);
        expect(backend.type).toBe(t);
      }
    });
  });

  describe('UMAP class backend option', () => {
    it('uses the specified backend when provided', async () => {
      // Test via the UMAP class
      const { UMAP } = await import('../umap');
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 5, backend: 'cpu' });

      const vectors = Array.from({ length: 10 }, (_, i) =>
        Array.from({ length: 4 }, (_, d) => Math.sin(i * 0.3 + d))
      );

      await umap.fit(vectors);
      expect(umap.activeBackend).toBe('cpu');
      expect(umap.embedding).toBeInstanceOf(Float32Array);
    });
  });
});
