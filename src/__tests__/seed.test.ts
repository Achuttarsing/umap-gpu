import { describe, it, expect } from 'vitest';
import { makeRng } from '../rng';
import { cpuSgd } from '../fallback/cpu-sgd';
import type { FuzzyGraph } from '../fuzzy-set';
import { computeEps } from './test-helpers';

function makeGraph(): FuzzyGraph {
  return {
    rows: new Uint32Array([0, 1, 2, 3]),
    cols: new Uint32Array([1, 0, 3, 2]),
    vals: new Float32Array([1.0, 1.0, 1.0, 1.0]),
    nVertices: 4,
  };
}

describe('makeRng', () => {
  it('returns identical sequences for the same seed', () => {
    const a = makeRng(42);
    const b = makeRng(42);
    for (let i = 0; i < 100; i++) {
      expect(a()).toBe(b());
    }
  });

  it('returns different sequences for different seeds', () => {
    const a = makeRng(1);
    const b = makeRng(2);
    let differs = false;
    for (let i = 0; i < 20; i++) {
      if (a() !== b()) { differs = true; break; }
    }
    expect(differs).toBe(true);
  });

  it('returns values in [0, 1)', () => {
    const rng = makeRng(0);
    for (let i = 0; i < 1000; i++) {
      const v = rng();
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThan(1);
    }
  });

  it('falls back to Math.random when no seed given', () => {
    const rng = makeRng();
    expect(rng).toBe(Math.random);
  });
});

describe('cpuSgd with seed', () => {
  const params = { a: 1.9292, b: 0.7915 };

  it('produces identical output for the same seed', () => {
    const graph = makeGraph();
    const eps = computeEps(graph.vals, 50);

    const embA = new Float32Array([5, 0, -5, 0, 5, 1, -5, 1]);
    const embB = new Float32Array(embA);
    cpuSgd(embA, graph, eps, 4, 2, 50, params, undefined, makeRng(42));
    cpuSgd(embB, graph, eps, 4, 2, 50, params, undefined, makeRng(42));

    expect(embA).toEqual(embB);
  });

  it('produces different output for different seeds', () => {
    const graph = makeGraph();
    const eps = computeEps(graph.vals, 50);

    const embA = new Float32Array([5, 0, -5, 0, 5, 1, -5, 1]);
    const embB = new Float32Array(embA);
    cpuSgd(embA, graph, eps, 4, 2, 50, params, undefined, makeRng(1));
    cpuSgd(embB, graph, eps, 4, 2, 50, params, undefined, makeRng(2));

    let differs = false;
    for (let i = 0; i < embA.length; i++) {
      if (embA[i] !== embB[i]) { differs = true; break; }
    }
    expect(differs).toBe(true);
  });
});
