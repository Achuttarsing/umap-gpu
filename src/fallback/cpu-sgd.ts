import type { FuzzyGraph } from '../fuzzy-set';

export interface CPUSgdParams {
  a: number;
  b: number;
  gamma?: number;
  negativeSampleRate?: number;
}

/**
 * CPU fallback SGD optimizer for environments without WebGPU.
 * Mirrors the GPU shader logic: per-edge attraction + negative-sample repulsion.
 */
export function cpuSgd(
  embedding: Float32Array,
  graph: FuzzyGraph,
  epochsPerSample: Float32Array,
  nVertices: number,
  nComponents: number,
  nEpochs: number,
  params: CPUSgdParams
): Float32Array {
  const { a, b, gamma = 1.0, negativeSampleRate = 5 } = params;
  const nEdges = graph.rows.length;

  const head = new Uint32Array(graph.rows);
  const tail = new Uint32Array(graph.cols);

  const epochOfNextSample = new Float32Array(nEdges).fill(0);
  const epochOfNextNegativeSample = new Float32Array(nEdges);
  for (let i = 0; i < nEdges; i++) {
    epochOfNextNegativeSample[i] = epochsPerSample[i] / negativeSampleRate;
  }

  function clip(v: number): number {
    return Math.max(-4.0, Math.min(4.0, v));
  }

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    const alpha = 1.0 - epoch / nEpochs;

    for (let edgeIdx = 0; edgeIdx < nEdges; edgeIdx++) {
      if (epochOfNextSample[edgeIdx] > epoch) continue;

      const i = head[edgeIdx];
      const j = tail[edgeIdx];

      // Attraction
      let distSq = 0;
      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i * nComponents + d] - embedding[j * nComponents + d];
        distSq += diff * diff;
      }

      const gradCoeffAttr =
        (-2.0 * a * b * Math.pow(distSq, b - 1.0)) /
        (a * Math.pow(distSq, b) + 1.0);

      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i * nComponents + d] - embedding[j * nComponents + d];
        const grad = clip(gradCoeffAttr * diff);
        embedding[i * nComponents + d] += alpha * grad;
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion (negative samples)
      const nNeg =
        epochOfNextNegativeSample[edgeIdx] > 0
          ? Math.floor(epochsPerSample[edgeIdx] / epochOfNextNegativeSample[edgeIdx])
          : 0;

      for (let s = 0; s < nNeg; s++) {
        const k = Math.floor(Math.random() * nVertices);
        if (k === i) continue;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i * nComponents + d] - embedding[k * nComponents + d];
          negDistSq += diff * diff;
        }

        const gradCoeffRep =
          (2.0 * gamma * b) /
          ((0.001 + negDistSq) * (a * Math.pow(negDistSq, b) + 1.0));

        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i * nComponents + d] - embedding[k * nComponents + d];
          const grad = clip(gradCoeffRep * diff);
          embedding[i * nComponents + d] += alpha * grad;
        }
      }

      epochOfNextNegativeSample[edgeIdx] +=
        epochsPerSample[edgeIdx] / negativeSampleRate;
    }
  }

  return embedding;
}
