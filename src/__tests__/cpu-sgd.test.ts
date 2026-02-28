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

  it('attraction updates both head node i and tail node j symmetrically', () => {
    // Single directed edge 0→1 with gamma=0 (no repulsion) so only
    // attraction runs. Old code updated only node i; the fix also updates j.
    const n = 2;
    const nComponents = 2;
    const embedding = new Float32Array([5, 0, -5, 0]); // node 0 at (5,0), node 1 at (-5,0)
    const graph: FuzzyGraph = {
      rows: new Float32Array([0]),
      cols: new Float32Array([1]),
      vals: new Float32Array([1.0]),
      nVertices: 2,
    };
    const eps = new Float32Array([1.0]); // sample once at epoch 0

    cpuSgd(embedding, graph, eps, n, nComponents, 1, { ...params, gamma: 0 });

    // Head node 0 must move toward tail node 1 (x decreases from 5)
    expect(embedding[0]).toBeLessThan(5);
    // Tail node 1 must also move toward head node 0 (x increases from -5)
    expect(embedding[2]).toBeGreaterThan(-5);
    // Both moves should be equal in magnitude (symmetric gradient)
    expect(Math.abs(embedding[0] - 5)).toBeCloseTo(Math.abs(embedding[2] - (-5)), 5);
  });
});
