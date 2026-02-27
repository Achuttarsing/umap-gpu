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
});
