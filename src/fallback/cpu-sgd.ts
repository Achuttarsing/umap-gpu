import type { FuzzyGraph } from '../fuzzy-set';

export interface CPUSgdParams {
  a: number;
  b: number;
  gamma?: number;
  negativeSampleRate?: number;
}

function clip(v: number): number {
  return Math.max(-4.0, Math.min(4.0, v));
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
  params: CPUSgdParams,
  onProgress?: (epoch: number, nEpochs: number) => void,
  rng: () => number = Math.random
): Float32Array {
  const { a, b, gamma = 1.0, negativeSampleRate = 5 } = params;
  const nEdges = graph.rows.length;

  const head = new Uint32Array(graph.rows);
  const tail = new Uint32Array(graph.cols);

  // Bug 4 fix: initialize to epochsPerSample (not 0), matching the Python
  // reference — epoch_of_next_sample = epochs_per_sample.copy() — so no edge
  // fires at epoch 0 (where alpha is at its maximum value of 1.0).
  const epochOfNextSample = new Float32Array(epochsPerSample);
  const epochOfNextNegativeSample = new Float32Array(nEdges);
  for (let i = 0; i < nEdges; i++) {
    epochOfNextNegativeSample[i] = epochsPerSample[i] / negativeSampleRate;
  }

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    onProgress?.(epoch, nEpochs);
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

      // Cache pow result: pow(distSq, b-1) = pow(distSq, b) / distSq
      const powB = Math.pow(distSq, b);
      const gradCoeffAttr =
        (-2.0 * a * b * (distSq > 0 ? powB / distSq : 0)) /
        (a * powB + 1.0);

      for (let d = 0; d < nComponents; d++) {
        const diff = embedding[i * nComponents + d] - embedding[j * nComponents + d];
        const grad = clip(gradCoeffAttr * diff);
        embedding[i * nComponents + d] += alpha * grad;
        embedding[j * nComponents + d] -= alpha * grad;
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion (negative samples) — matches umap-js reference formula
      const epochsPerNeg = epochsPerSample[edgeIdx] / negativeSampleRate;
      const nNeg = Math.max(0, Math.floor(
        (epoch - epochOfNextNegativeSample[edgeIdx]) / epochsPerNeg
      ));
      epochOfNextNegativeSample[edgeIdx] += nNeg * epochsPerNeg;

      for (let s = 0; s < nNeg; s++) {
        const k = Math.floor(rng() * nVertices);
        if (k === i) continue;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i * nComponents + d] - embedding[k * nComponents + d];
          negDistSq += diff * diff;
        }

        const negPowB = Math.pow(negDistSq, b);
        const gradCoeffRep =
          (2.0 * gamma * b) /
          ((0.001 + negDistSq) * (a * negPowB + 1.0));

        for (let d = 0; d < nComponents; d++) {
          const diff = embedding[i * nComponents + d] - embedding[k * nComponents + d];
          const grad = clip(gradCoeffRep * diff);
          embedding[i * nComponents + d] += alpha * grad;
        }
      }
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
  params: CPUSgdParams,
  onProgress?: (epoch: number, nEpochs: number) => void,
  rng: () => number = Math.random
): Float32Array {
  const { a, b, gamma = 1.0, negativeSampleRate = 5 } = params;
  const nEdges = graph.rows.length;

  const head = new Uint32Array(graph.rows);  // new-point indices
  const tail = new Uint32Array(graph.cols);  // training-point indices

  // Bug 4 fix: initialize to epochsPerSample (not 0), matching the Python reference.
  const epochOfNextSample = new Float32Array(epochsPerSample);
  const epochOfNextNegativeSample = new Float32Array(nEdges);
  for (let i = 0; i < nEdges; i++) {
    epochOfNextNegativeSample[i] = epochsPerSample[i] / negativeSampleRate;
  }

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    onProgress?.(epoch, nEpochs);
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

      // Cache pow result: pow(distSq, b-1) = pow(distSq, b) / distSq
      const powB = Math.pow(distSq, b);
      const gradCoeffAttr =
        (-2.0 * a * b * (distSq > 0 ? powB / distSq : 0)) /
        (a * powB + 1.0);

      for (let d = 0; d < nComponents; d++) {
        const diff = embeddingNew[i * nComponents + d] - embeddingTrain[j * nComponents + d];
        embeddingNew[i * nComponents + d] += alpha * clip(gradCoeffAttr * diff);
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion: push new point away from random training points — matches umap-js reference formula
      const epochsPerNeg = epochsPerSample[edgeIdx] / negativeSampleRate;
      const nNeg = Math.max(0, Math.floor(
        (epoch - epochOfNextNegativeSample[edgeIdx]) / epochsPerNeg
      ));
      epochOfNextNegativeSample[edgeIdx] += nNeg * epochsPerNeg;

      for (let s = 0; s < nNeg; s++) {
        const k = Math.floor(rng() * nTrain);
        if (k === j) continue;

        let negDistSq = 0;
        for (let d = 0; d < nComponents; d++) {
          const diff = embeddingNew[i * nComponents + d] - embeddingTrain[k * nComponents + d];
          negDistSq += diff * diff;
        }

        const negPowB = Math.pow(negDistSq, b);
        const gradCoeffRep =
          (2.0 * gamma * b) /
          ((0.001 + negDistSq) * (a * negPowB + 1.0));

        for (let d = 0; d < nComponents; d++) {
          const diff = embeddingNew[i * nComponents + d] - embeddingTrain[k * nComponents + d];
          embeddingNew[i * nComponents + d] += alpha * clip(gradCoeffRep * diff);
        }
      }
    }
  }

  return embeddingNew;
}
