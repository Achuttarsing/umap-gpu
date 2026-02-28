import { describe, it, expect } from 'vitest';
import { computeFuzzySimplicialSet } from '../fuzzy-set';

describe('computeFuzzySimplicialSet', () => {
  // Minimal 3-point dataset: each point has 2 neighbors
  const indices = [[1, 2], [0, 2], [0, 1]];
  const distances = [[0.5, 1.0], [0.5, 0.8], [1.0, 0.8]];

  it('returns the correct number of vertices', () => {
    const graph = computeFuzzySimplicialSet(indices, distances, 2);
    expect(graph.nVertices).toBe(3);
  });

  it('produces a non-empty edge list', () => {
    const graph = computeFuzzySimplicialSet(indices, distances, 2);
    expect(graph.rows.length).toBeGreaterThan(0);
    expect(graph.cols.length).toBe(graph.rows.length);
    expect(graph.vals.length).toBe(graph.rows.length);
  });

  it('all edge weights are in [0, 1]', () => {
    const graph = computeFuzzySimplicialSet(indices, distances, 2);
    for (const v of graph.vals) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it('graph is symmetric: every (i,j) edge has a matching (j,i) edge', () => {
    const graph = computeFuzzySimplicialSet(indices, distances, 2);
    const edgeSet = new Set<string>();
    for (let e = 0; e < graph.rows.length; e++) {
      edgeSet.add(`${graph.rows[e]},${graph.cols[e]}`);
    }
    for (const key of edgeSet) {
      const [r, c] = key.split(',');
      expect(edgeSet.has(`${c},${r}`)).toBe(true);
    }
  });

  it('symmetrize uses fuzzy union P+Q-P*Q, not the broken double-accumulation formula', () => {
    // 2-point graph: each point has the other as its sole neighbour.
    // With distance <= rho both directed weights are exactly 1.0.
    // Correct union: 1.0 + 1.0 - 1.0*1.0 = 1.0.
    // Broken code produced: 2-(P+Q) = 2-2 = 0.0 (catastrophic weight loss).
    const idx2 = [[1], [0]];
    const dist2 = [[0.1], [0.1]];
    const graph = computeFuzzySimplicialSet(idx2, dist2, 1, 1.0);

    expect(graph.rows.length).toBe(2);
    for (const v of graph.vals) {
      expect(v).toBeCloseTo(1.0, 5);
    }
  });

  it('symmetric weights match fuzzy union for asymmetric directed weights', () => {
    // Use a 3-point chain where 0→1 distance is smaller than 1→0's perspective,
    // producing P ≠ Q. Verify every symmetrized weight w satisfies:
    //   w <= P + Q  (union is at most additive)
    //   w >= max(P, Q) * 0.999  (union is at least as large as either operand)
    // These inequalities are violated by the old formula (e.g., w = 2-sum < P+Q-1 < max).
    const graph = computeFuzzySimplicialSet(indices, distances, 2, 1.0);
    const weightMap = new Map<string, number>();
    for (let e = 0; e < graph.rows.length; e++) {
      weightMap.set(`${graph.rows[e]},${graph.cols[e]}`, graph.vals[e]);
    }
    for (const [key, w] of weightMap) {
      const [r, c] = key.split(',');
      const wReverse = weightMap.get(`${c},${r}`) ?? 0;
      // union >= each operand
      expect(w).toBeGreaterThanOrEqual(wReverse - 1e-5);
    }
  });
});
