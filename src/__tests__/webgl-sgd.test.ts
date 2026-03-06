/**
 * Tests for the WebGL SGD optimizer.
 *
 * Since WebGL 2 (OffscreenCanvas) is not available in Node.js test environments,
 * these tests verify the algorithm correctness by instantiating the WebGLSgd class
 * directly with a mock init, exercising the same SGD code path as the CPU fallback.
 */

import { describe, it, expect, vi } from 'vitest';

vi.mock('../hnsw-knn', () => ({
  computeKNN: vi.fn(),
  computeKNNWithIndex: vi.fn(),
}));

import { computeFuzzySimplicialSet } from '../fuzzy-set';
import { computeEpochsPerSample } from '../umap';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

// ─── Inline WebGL SGD to test the algorithm directly ────────────────────────

// We can't use WebGLSgd.init() in Node (no WebGL 2), so we import the class
// and bypass the init check by setting the `available` flag manually.
import { WebGLSgd } from '../webgl/sgd';

function makeWebGLSgd(): WebGLSgd {
  const sgd = new WebGLSgd();
  // Bypass WebGL 2 context check for Node.js test environment
  (sgd as unknown as { available: boolean }).available = true;
  return sgd;
}

function makeSimpleGraph(): FuzzyGraph {
  return {
    rows: new Uint32Array([0, 1]),
    cols: new Uint32Array([1, 0]),
    vals: new Float32Array([1.0, 1.0]),
    nVertices: 2,
  };
}

const UMAP_PARAMS = { a: 1.9292, b: 0.7915, gamma: 1.0, negativeSampleRate: 5 };

describe('WebGLSgd', () => {
  it('returns a Float32Array of correct length', async () => {
    const sgd = makeWebGLSgd();
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 10);
    const embedding = new Float32Array([1, 0, -1, 0]);

    const result = await sgd.optimize(embedding, graph, eps, 2, 2, 10, UMAP_PARAMS);
    expect(result).toBeInstanceOf(Float32Array);
    expect(result.length).toBe(4);
  });

  it('modifies the embedding after optimization', async () => {
    const sgd = makeWebGLSgd();
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 50);
    const initial = new Float32Array([5, 0, -5, 0]);
    const embedding = new Float32Array(initial);

    await sgd.optimize(embedding, graph, eps, 2, 2, 50, UMAP_PARAMS);

    let changed = false;
    for (let i = 0; i < embedding.length; i++) {
      if (Math.abs(embedding[i] - initial[i]) > 1e-6) { changed = true; break; }
    }
    expect(changed).toBe(true);
  });

  it('attraction updates both nodes symmetrically', async () => {
    const sgd = makeWebGLSgd();
    const graph: FuzzyGraph = {
      rows: new Uint32Array([0]),
      cols: new Uint32Array([1]),
      vals: new Float32Array([1.0]),
      nVertices: 2,
    };
    const eps = new Float32Array([1.0]);
    const embedding = new Float32Array([5, 0, -5, 0]);

    await sgd.optimize(embedding, graph, eps, 2, 2, 2, { ...UMAP_PARAMS, gamma: 0 });

    // Head node 0 must move toward tail node 1 (x decreases from 5)
    expect(embedding[0]).toBeLessThan(5);
    // Tail node 1 must also move toward head node 0 (x increases from -5)
    expect(embedding[2]).toBeGreaterThan(-5);
    // Both moves should be equal in magnitude
    expect(Math.abs(embedding[0] - 5)).toBeCloseTo(Math.abs(embedding[2] - (-5)), 5);
  });

  it('edges do not fire at epoch 0 (Bug 4 fix)', async () => {
    const sgd = makeWebGLSgd();
    const graph = computeFuzzySimplicialSet([[1], [0]], [[0.5], [0.5]], 1);
    const eps = computeEpochsPerSample(graph.vals, 100);
    const initial = new Float32Array([5, 0, -5, 0]);
    const before = new Float32Array(initial);

    await sgd.optimize(initial, graph, eps, 2, 2, 1, UMAP_PARAMS);

    for (let i = 0; i < before.length; i++) {
      expect(initial[i]).toBe(before[i]);
    }
  });

  it('invokes progress callback for each epoch', async () => {
    const sgd = makeWebGLSgd();
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 5);
    const embedding = new Float32Array([1, 0, -1, 0]);
    const calls: Array<[number, number]> = [];

    await sgd.optimize(embedding, graph, eps, 2, 2, 5, UMAP_PARAMS, (epoch, total) => {
      calls.push([epoch, total]);
    });

    expect(calls.length).toBe(5);
    expect(calls[0]).toEqual([0, 5]);
    expect(calls[4]).toEqual([4, 5]);
  });

  it('throws if optimize is called before init', async () => {
    const sgd = new WebGLSgd();
    const graph = makeSimpleGraph();
    const eps = computeEps(graph.vals, 5);
    const embedding = new Float32Array([1, 0, -1, 0]);

    await expect(sgd.optimize(embedding, graph, eps, 2, 2, 5, UMAP_PARAMS))
      .rejects.toThrow('WebGLSgd.init() must be called first');
  });
});
