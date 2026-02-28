import { describe, it, expect } from 'vitest';
import { cpuSgd } from '../fallback/cpu-sgd';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

function makeSimpleGraph(): FuzzyGraph {
  return {
    rows: new Float32Array([0, 1]),
    cols: new Float32Array([1, 0]),
    vals: new Float32Array([1.0, 1.0]),
    nVertices: 2,
  };
}

describe('cpuSgd', () => {
  const params = { a: 1.9292, b: 0.7915 };

  it('returns a Float32Array of length n * nComponents', () => {
    const n = 2;
    const nComponents = 2;
    const embedding = new Float32Array([1, 0, -1, 0]);
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 10);

    const result = cpuSgd(embedding, graph, eps, n, nComponents, 10, params);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(n * nComponents);
  });

  it('modifies the embedding after optimization', () => {
    const n = 2;
    const nComponents = 2;
    const initial = new Float32Array([5, 0, -5, 0]);
    const embedding = new Float32Array(initial);
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 50);

    cpuSgd(embedding, graph, eps, n, nComponents, 50, params);

    // At least one coordinate should have moved
    let changed = false;
    for (let i = 0; i < embedding.length; i++) {
      if (Math.abs(embedding[i] - initial[i]) > 1e-6) { changed = true; break; }
    }
    expect(changed).toBe(true);
  });
});
