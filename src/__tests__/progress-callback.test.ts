import { describe, it, expect, vi } from 'vitest';
import { cpuSgd, cpuSgdTransform } from '../fallback/cpu-sgd';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

// Mock hnswlib-wasm so we can test UMAP class with synthetic k-NN results.
vi.mock('../hnsw-knn', () => {
  const fakeIndex = {
    searchKnn: (queryVectors: number[][], nNeighbors: number) => {
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

import { UMAP, fit } from '../umap';
import type { ProgressEvent } from '../umap';

function makeVectors(n: number, dim: number): number[][] {
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: dim }, (_, d) => Math.sin(i * 0.3 + d))
  );
}

function makeSimpleGraph(): FuzzyGraph {
  return {
    rows: new Float32Array([0, 1]),
    cols: new Float32Array([1, 0]),
    vals: new Float32Array([1.0, 1.0]),
    nVertices: 2,
  };
}

const sgdParams = { a: 1.9292, b: 0.7915 };
const nEpochs = 5;

// ─── cpuSgd ──────────────────────────────────────────────────────────────────

describe('cpuSgd onEpoch callback', () => {
  it('is called once per epoch', () => {
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, nEpochs);
    const embedding = new Float32Array([1, 0, -1, 0]);
    const onEpoch = vi.fn();

    cpuSgd(embedding, graph, eps, 2, 2, nEpochs, sgdParams, onEpoch);

    expect(onEpoch).toHaveBeenCalledTimes(nEpochs);
  });

  it('receives correct (epoch, nEpochs) arguments', () => {
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, nEpochs);
    const embedding = new Float32Array([1, 0, -1, 0]);
    const calls: [number, number][] = [];

    cpuSgd(embedding, graph, eps, 2, 2, nEpochs, sgdParams, (epoch, total) => {
      calls.push([epoch, total]);
    });

    expect(calls).toHaveLength(nEpochs);
    calls.forEach(([epoch, total], i) => {
      expect(epoch).toBe(i);
      expect(total).toBe(nEpochs);
    });
  });

  it('is not required (omitting it does not throw)', () => {
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, nEpochs);
    const embedding = new Float32Array([1, 0, -1, 0]);
    expect(() =>
      cpuSgd(embedding, graph, eps, 2, 2, nEpochs, sgdParams)
    ).not.toThrow();
  });
});

// ─── cpuSgdTransform ─────────────────────────────────────────────────────────

describe('cpuSgdTransform onEpoch callback', () => {
  it('is called once per epoch', () => {
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, nEpochs);
    const embeddingNew = new Float32Array([1, 0]);
    const embeddingTrain = new Float32Array([0, 0, -1, 0]);
    const onEpoch = vi.fn();

    cpuSgdTransform(embeddingNew, embeddingTrain, graph, eps, 1, 2, 2, nEpochs, sgdParams, onEpoch);

    expect(onEpoch).toHaveBeenCalledTimes(nEpochs);
  });

  it('receives correct (epoch, nEpochs) arguments', () => {
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, nEpochs);
    const embeddingNew = new Float32Array([1, 0]);
    const embeddingTrain = new Float32Array([0, 0, -1, 0]);
    const calls: [number, number][] = [];

    cpuSgdTransform(
      embeddingNew, embeddingTrain, graph, eps, 1, 2, 2, nEpochs, sgdParams,
      (epoch, total) => calls.push([epoch, total])
    );

    expect(calls).toHaveLength(nEpochs);
    calls.forEach(([epoch, total], i) => {
      expect(epoch).toBe(i);
      expect(total).toBe(nEpochs);
    });
  });
});

// ─── UMAP class ──────────────────────────────────────────────────────────────

describe('UMAP class progress callback', () => {
  describe('fit()', () => {
    it('calls onProgress for knn and fuzzy-set stages', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 3 });
      const events: ProgressEvent[] = [];
      await umap.fit(makeVectors(8, 4), (e) => events.push(e));

      const stages = events.map((e) => e.stage);
      expect(stages).toContain('knn');
      expect(stages).toContain('fuzzy-set');
    });

    it('calls onProgress with sgd stage events for each epoch', async () => {
      const nEpochs = 4;
      const umap = new UMAP({ nNeighbors: 3, nEpochs });
      const sgdEvents: ProgressEvent[] = [];

      await umap.fit(makeVectors(8, 4), (e) => {
        if (e.stage === 'sgd') sgdEvents.push(e);
      });

      expect(sgdEvents).toHaveLength(nEpochs);
      sgdEvents.forEach((e, i) => {
        expect(e.epoch).toBe(i);
        expect(e.nEpochs).toBe(nEpochs);
      });
    });

    it('sgd events arrive in epoch order', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 6 });
      const epochs: number[] = [];

      await umap.fit(makeVectors(8, 4), (e) => {
        if (e.stage === 'sgd') epochs.push(e.epoch!);
      });

      for (let i = 0; i < epochs.length - 1; i++) {
        expect(epochs[i + 1]).toBe(epochs[i] + 1);
      }
    });

    it('knn is reported before fuzzy-set', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 3 });
      const stages: string[] = [];
      await umap.fit(makeVectors(8, 4), (e) => stages.push(e.stage));

      expect(stages.indexOf('knn')).toBeLessThan(stages.indexOf('fuzzy-set'));
    });

    it('works without a callback (no second argument)', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 3 });
      await expect(umap.fit(makeVectors(6, 4))).resolves.toBe(umap);
    });
  });

  describe('fit_transform()', () => {
    it('forwards onProgress to fit()', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 3 });
      const events: ProgressEvent[] = [];
      await umap.fit_transform(makeVectors(8, 4), (e) => events.push(e));

      const stages = events.map((e) => e.stage);
      expect(stages).toContain('knn');
      expect(stages).toContain('fuzzy-set');
      expect(stages).toContain('sgd');
    });

    it('works without a callback', async () => {
      const umap = new UMAP({ nNeighbors: 3, nEpochs: 3 });
      const result = await umap.fit_transform(makeVectors(6, 4));
      expect(result).toBeInstanceOf(Float32Array);
    });
  });
});

// ─── Functional fit() ────────────────────────────────────────────────────────

describe('functional fit() onProgress option', () => {
  it('calls onProgress for knn, fuzzy-set and sgd stages', async () => {
    const events: ProgressEvent[] = [];
    await fit(makeVectors(8, 4), {
      nNeighbors: 3,
      nEpochs: 3,
      onProgress: (e) => events.push(e),
    });

    const stages = new Set(events.map((e) => e.stage));
    expect(stages.has('knn')).toBe(true);
    expect(stages.has('fuzzy-set')).toBe(true);
    expect(stages.has('sgd')).toBe(true);
  });

  it('sgd events carry correct epoch and nEpochs', async () => {
    const nEpochs = 4;
    const sgdEvents: ProgressEvent[] = [];

    await fit(makeVectors(8, 4), {
      nNeighbors: 3,
      nEpochs,
      onProgress: (e) => { if (e.stage === 'sgd') sgdEvents.push(e); },
    });

    expect(sgdEvents).toHaveLength(nEpochs);
    sgdEvents.forEach((e, i) => {
      expect(e.epoch).toBe(i);
      expect(e.nEpochs).toBe(nEpochs);
    });
  });

  it('works without onProgress option', async () => {
    const result = await fit(makeVectors(6, 4), { nNeighbors: 3, nEpochs: 3 });
    expect(result).toBeInstanceOf(Float32Array);
  });
});
