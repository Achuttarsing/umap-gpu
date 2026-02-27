import { describe, it, expect, vi } from 'vitest';

// hnsw-knn depends on hnswlib-wasm (WASM) which isn't available in Node.
// Mock it so we can import the pure-math helpers from umap.ts.
vi.mock('../hnsw-knn', () => ({ computeKNN: vi.fn() }));

import { findAB, computeEpochsPerSample } from '../umap';

describe('findAB', () => {
  it('returns pre-fitted values for minDist=0.1, spread=1.0', () => {
    const { a, b } = findAB(0.1, 1.0);
    expect(a).toBeCloseTo(1.9292, 3);
    expect(b).toBeCloseTo(0.7915, 3);
  });

  it('returns pre-fitted values for minDist=0.0, spread=1.0', () => {
    const { a, b } = findAB(0.0, 1.0);
    expect(a).toBeCloseTo(1.8956, 3);
    expect(b).toBeCloseTo(0.8006, 3);
  });

  it('returns pre-fitted values for minDist=0.5, spread=1.0', () => {
    const { a, b } = findAB(0.5, 1.0);
    expect(a).toBeCloseTo(1.5769, 3);
    expect(b).toBeCloseTo(0.8951, 3);
  });

  it('returns positive a and b for arbitrary parameters', () => {
    const { a, b } = findAB(0.25, 2.0);
    expect(a).toBeGreaterThan(0);
    expect(b).toBeGreaterThan(0);
  });
});

describe('computeEpochsPerSample', () => {
  it('assigns nEpochs to the max-weight edge', () => {
    const weights = new Float32Array([0.5, 1.0, 0.25]);
    const result = computeEpochsPerSample(weights, 100);
    // max weight (1.0) → 100 / 1.0 = 100
    expect(result[1]).toBeCloseTo(100);
  });

  it('assigns proportional periods to lower-weight edges', () => {
    const weights = new Float32Array([0.5, 1.0]);
    const result = computeEpochsPerSample(weights, 200);
    // weight 0.5 → normalized 0.5 → 200 / 0.5 = 400
    expect(result[0]).toBeCloseTo(400);
    expect(result[1]).toBeCloseTo(200);
  });

  it('assigns -1 to zero-weight edges', () => {
    const weights = new Float32Array([0.0, 1.0]);
    const result = computeEpochsPerSample(weights, 100);
    expect(result[0]).toBe(-1);
  });
});
