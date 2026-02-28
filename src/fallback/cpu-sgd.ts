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

/**
 * CPU SGD for UMAP.transform(): optimizes only the new-point embeddings.
 * The training embedding is read-only; attraction pulls new points toward
 * their training neighbors, and repulsion pushes them away from random
 * training points.
 *
 * @param embeddingNew   - New-point embeddings to optimize [nNew × nComponents]
 * @param embeddingTrain - Fixed training embeddings [nTrain × nComponents]
 * @param graph          - Bipartite graph: rows=new-point indices, cols=training-point indices
 * @param epochsPerSample - Per-edge epoch sampling schedule
 * @param nNew           - Number of new points
 * @param nTrain         - Number of training points
 * @param nComponents    - Embedding dimensionality
 * @param nEpochs        - Number of optimization epochs
 * @param params         - UMAP curve parameters
 */
export function cpuSgdTransform(
  embeddingNew: Float32Array,
  embeddingTrain: Float32Array,
  graph: FuzzyGraph,
  epochsPerSample: Float32Array,
  nNew: number,
  nTrain: number,
  nComponents: number,
  nEpochs: number,
  params: CPUSgdParams
): Float32Array {
  const { a, b, gamma = 1.0, negativeSampleRate = 5 } = params;
  const nEdges = graph.rows.length;

  const head = new Uint32Array(graph.rows);  // new-point indices
  const tail = new Uint32Array(graph.cols);  // training-point indices

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

      const i = head[edgeIdx];  // new point
      const j = tail[edgeIdx];  // training neighbor

      // Attraction: pull new point toward training neighbor (fixed)
      let distSq = 0;
      for (let d = 0; d < nComponents; d++) {
        const diff = embeddingNew[i * nComponents + d] - embeddingTrain[j * nComponents + d];
        distSq += diff * diff;
      }

      const gradCoeffAttr =
        (-2.0 * a * b * Math.pow(distSq, b - 1.0)) /
        (a * Math.pow(distSq, b) + 1.0);

      for (let d = 0; d < nComponents; d++) {
        const diff = embeddingNew[i * nComponents + d] - embeddingTrain[j * nComponents + d];
        embeddingNew[i * nComponents + d] += alpha * clip(gradCoeffAttr * diff);
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion: push new point away from random training points
      const nNeg =
        epochOfNextNegativeSample[edgeIdx] > 0
          ? Math.floor(epochsPerSample[edgeIdx] / epochOfNextNegativeSample[edgeIdx])
          : 0;

      for (let s = 0; s < nNeg; s++) {
        const k = Math.floor(Math.random() * nTrain);
        if (k === j) continue;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embeddingNew[i * nComponents + d] - embeddingTrain[k * nComponents + d];
          negDistSq += diff * diff;
        }

        const gradCoeffRep =
          (2.0 * gamma * b) /
          ((0.001 + negDistSq) * (a * Math.pow(negDistSq, b) + 1.0));

        for (let d = 0; d < nComponents; d++) {
          const diff = embeddingNew[i * nComponents + d] - embeddingTrain[k * nComponents + d];
          embeddingNew[i * nComponents + d] += alpha * clip(gradCoeffRep * diff);
        }
      }

      epochOfNextNegativeSample[edgeIdx] +=
        epochsPerSample[edgeIdx] / negativeSampleRate;
    }
  }

  return embeddingNew;
}
