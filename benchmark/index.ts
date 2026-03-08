/**
 * UMAP Benchmark: CPU vs GPU SGD
 *
 * Measures full pipeline wall time (kNN + fuzzy set + SGD) for both backends
 * across a range of dataset sizes.
 *
 * Note: hnswlib-wasm is web-only WASM and cannot run in Node. We use an
 * O(n²) brute-force kNN here — both backends share the same pre-computed
 * graph, so the comparison focuses on SGD performance.
 */
import { describe, it, beforeAll } from 'vitest';
import { computeFuzzySimplicialSet } from '../src/fuzzy-set';
import { computeEpochsPerSample } from '../src/umap';
import { cpuSgd } from '../src/fallback/cpu-sgd';
import { GPUSgd } from '../src/gpu/sgd';
import { findAB } from '../src/umap';

const SIZES = [100, 500, 1_000, 5_000, 10_000];
const DIM = 50;
const N_NEIGHBORS = 15;
const N_EPOCHS = 200;
const { a, b } = findAB(0.1, 1.0);
const SGD_PARAMS = { a, b, gamma: 1.0, negativeSampleRate: 5 };

function makeVectors(n: number, dim: number): number[][] {
  const vecs: number[][] = [];
  for (let i = 0; i < n; i++) {
    const v: number[] = [];
    for (let d = 0; d < dim; d++) v.push(Math.random());
    vecs.push(v);
  }
  return vecs;
}

function bruteForceKNN(
  vectors: number[][],
  k: number
): { indices: number[][]; distances: number[][] } {
  const n = vectors.length;
  const dim = vectors[0].length;
  const outIdx: number[][] = [];
  const outDist: number[][] = [];
  for (let i = 0; i < n; i++) {
    const dists: Array<{ idx: number; d: number }> = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      let d = 0;
      for (let dd = 0; dd < dim; dd++) {
        const diff = vectors[i][dd] - vectors[j][dd];
        d += diff * diff;
      }
      dists.push({ idx: j, d: Math.sqrt(d) });
    }
    dists.sort((a, b) => a.d - b.d);
    outIdx.push(dists.slice(0, k).map((x) => x.idx));
    outDist.push(dists.slice(0, k).map((x) => x.d));
  }
  return { indices: outIdx, distances: outDist };
}

function makeInitialEmbedding(n: number): Float32Array {
  const emb = new Float32Array(n * 2);
  for (let i = 0; i < emb.length; i++) emb[i] = Math.random() * 20 - 10;
  return emb;
}

const cpuResults: { n: number; ms: number }[] = [];
const gpuResults: { n: number; ms: number }[] = [];

describe.sequential('CPU', () => {
  for (const n of SIZES) {
    it(`n=${n}`, () => {
      const vectors = makeVectors(n, DIM);
      const t0 = performance.now();

      const { indices, distances } = bruteForceKNN(vectors, N_NEIGHBORS);
      const graph = computeFuzzySimplicialSet(indices, distances, N_NEIGHBORS);
      const epochsPerSample = computeEpochsPerSample(graph.vals, N_EPOCHS);
      const embedding = makeInitialEmbedding(n);
      cpuSgd(embedding, graph, epochsPerSample, n, 2, N_EPOCHS, { a, b });

      const ms = Math.round(performance.now() - t0);
      cpuResults.push({ n, ms });
      console.log(`  CPU n=${n}: ${ms}ms`);
    });
  }
});

describe.sequential('GPU', () => {
  beforeAll(async () => {
    const { create } = await import('webgpu');
    const gpu = create([]);
    if (typeof globalThis.navigator === 'undefined') {
      Object.defineProperty(globalThis, 'navigator', {
        value: { gpu },
        configurable: true,
        writable: true,
      });
    } else {
      (globalThis.navigator as { gpu?: unknown }).gpu = gpu;
    }
  });

  for (const n of SIZES) {
    it(`n=${n}`, async (ctx) => {
      if (!(globalThis as Record<string, unknown>).__webGPUAvailable) {
        ctx.skip();
        return;
      }
      const vectors = makeVectors(n, DIM);
      const t0 = performance.now();

      const { indices, distances } = bruteForceKNN(vectors, N_NEIGHBORS);
      const graph = computeFuzzySimplicialSet(indices, distances, N_NEIGHBORS);
      const epochsPerSample = computeEpochsPerSample(graph.vals, N_EPOCHS);
      const embedding = makeInitialEmbedding(n);

      const gpuSgd = new GPUSgd();
      await gpuSgd.init();
      await gpuSgd.optimize(
        embedding,
        new Uint32Array(graph.rows),
        new Uint32Array(graph.cols),
        epochsPerSample,
        n,
        2,
        N_EPOCHS,
        SGD_PARAMS
      );

      const ms = Math.round(performance.now() - t0);
      gpuResults.push({ n, ms });
      console.log(`  GPU n=${n}: ${ms}ms`);
    });
  }
});

describe('Summary', () => {
  it('print table', () => {
    console.log('\n=== UMAP Benchmark Results ===\n');
    const header = `${'n'.padStart(8)}  ${'CPU (ms)'.padStart(10)}  ${'GPU (ms)'.padStart(10)}  ${'Speedup'.padStart(10)}`;
    const sep = '─'.repeat(header.length);
    console.log(header);
    console.log(sep);

    const gpuMap = new Map(gpuResults.map((r) => [r.n, r.ms]));

    for (const { n, ms: cpuMs } of cpuResults) {
      const gpuMs = gpuMap.get(n);
      const speedup = gpuMs != null ? (cpuMs / gpuMs).toFixed(2) + 'x' : 'N/A (skipped)';
      const gpuCol = gpuMs != null ? String(gpuMs).padStart(10) : '   skipped'.padStart(10);
      console.log(
        `${String(n).padStart(8)}  ${String(cpuMs).padStart(10)}  ${gpuCol}  ${speedup.padStart(10)}`
      );
    }

    if (cpuResults.length === 0) {
      console.log('  (no results collected)');
    }
  });
});
