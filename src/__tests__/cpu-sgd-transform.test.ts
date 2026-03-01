import { describe, it, expect } from 'vitest';
import { cpuSgdTransform } from '../fallback/cpu-sgd';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

/** Bipartite graph: new point 0 → training point 0, new point 1 → training point 1 */
function makeBipartiteGraph(): FuzzyGraph {
  return {
    rows: new Uint32Array([0, 1]),   // new-point indices
    cols: new Uint32Array([0, 1]),   // training-point indices
    vals: new Float32Array([1.0, 1.0]),
    nVertices: 2,
  };
}

const params = { a: 1.9292, b: 0.7915 };

describe('cpuSgdTransform', () => {
  it('returns a Float32Array of length nNew * nComponents', () => {
    const embeddingNew = new Float32Array([5, 0, -5, 0]);   // 2 new pts × 2D
    const embeddingTrain = new Float32Array([1, 0, -1, 0]); // 2 train pts × 2D
    const graph = makeBipartiteGraph();
    const eps = computeEps(graph.vals, 10);

    const result = cpuSgdTransform(
      embeddingNew, embeddingTrain, graph, eps, 2, 2, 2, 10, params
    );

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(4); // 2 pts × 2 dims
  });

  it('moves new-point embeddings toward their training neighbors', () => {
    // New points start far from their training counterparts
    const embeddingNew = new Float32Array([10, 0, -10, 0]);
    const embeddingTrain = new Float32Array([0, 0, 0, 0]);
    const graph = makeBipartiteGraph();
    const eps = computeEps(graph.vals, 100);

    const before0 = embeddingNew[0]; // 10
    cpuSgdTransform(embeddingNew, embeddingTrain, graph, eps, 2, 2, 2, 100, params);

    // New point 0 should have moved closer to training point 0 (at 0, 0)
    expect(Math.abs(embeddingNew[0])).toBeLessThan(Math.abs(before0));
  });

  it('does not modify the training embedding', () => {
    const embeddingNew = new Float32Array([5, 0, -5, 0]);
    const embeddingTrain = new Float32Array([1, 0, -1, 0]);
    const trainCopy = new Float32Array(embeddingTrain);
    const graph = makeBipartiteGraph();
    const eps = computeEps(graph.vals, 50);

    cpuSgdTransform(embeddingNew, embeddingTrain, graph, eps, 2, 2, 2, 50, params);

    for (let i = 0; i < trainCopy.length; i++) {
      expect(embeddingTrain[i]).toBe(trainCopy[i]);
    }
  });

  it('works with nComponents = 1', () => {
    const embeddingNew = new Float32Array([8, -8]);
    const embeddingTrain = new Float32Array([0, 0]);
    const graph = makeBipartiteGraph();
    const eps = computeEps(graph.vals, 30);

    const result = cpuSgdTransform(
      embeddingNew, embeddingTrain, graph, eps, 2, 2, 1, 30, params
    );

    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(2);
  });
});
