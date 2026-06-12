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
 *
 * Mirrors the reference (and the GPU two-pass) implementation:
 *   - each active edge contributes one attraction step plus exactly
 *     `negativeSampleRate` repulsion steps against random vertices;
 *   - squared distances are floored at 1e-6;
 *   - gradients are accumulated into a forces buffer from the *current*
 *     embedding (Jacobi update) and applied once per epoch — matching the GPU
 *     path rather than the sequential (Gauss-Seidel) order.
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
  const nc = nComponents;
  const nEdges = graph.rows.length;

  const head = graph.rows;
  const tail = graph.cols;

  // Bug 4 fix: initialize to epochsPerSample (not 0), matching the reference —
  // epoch_of_next_sample = epochs_per_sample.copy() — so no edge fires at
  // epoch 0 (where alpha is at its maximum value of 1.0).
  const epochOfNextSample = new Float32Array(epochsPerSample);

  // Forces accumulator (Jacobi): filled from the current embedding each epoch,
  // then applied so every edge sees the same positions within an epoch.
  const forces = new Float32Array(nVertices * nc);

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    onProgress?.(epoch, nEpochs);
    const alpha = 1.0 - epoch / nEpochs;
    forces.fill(0);

    for (let edgeIdx = 0; edgeIdx < nEdges; edgeIdx++) {
      if (epochOfNextSample[edgeIdx] > epoch) continue;

      const i = head[edgeIdx];
      const j = tail[edgeIdx];

      // Attraction
      let distSq = 0;
      for (let d = 0; d < nc; d++) {
        const diff = embedding[i * nc + d] - embedding[j * nc + d];
        distSq += diff * diff;
      }
      distSq = Math.max(distSq, 1e-6);

      // grad_coeff = -2ab * distSq^(b-1) / (1 + a*distSq^b)
      const powB = Math.pow(distSq, b);
      const gradCoeffAttr = (-2.0 * a * b * (powB / distSq)) / (a * powB + 1.0);

      for (let d = 0; d < nc; d++) {
        const diff = embedding[i * nc + d] - embedding[j * nc + d];
        const grad = clip(gradCoeffAttr * diff);
        forces[i * nc + d] += grad;
        forces[j * nc + d] -= grad;
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion: a fixed number of negative samples per active edge.
      for (let s = 0; s < negativeSampleRate; s++) {
        const kk = Math.floor(rng() * nVertices);

        let negDistSq = 0;
        for (let d = 0; d < nc; d++) {
          const diff = embedding[i * nc + d] - embedding[kk * nc + d];
          negDistSq += diff * diff;
        }
        negDistSq = Math.max(negDistSq, 1e-6);

        const negPowB = Math.pow(negDistSq, b);
        const gradCoeffRep =
          (2.0 * gamma * b) / ((0.001 + negDistSq) * (a * negPowB + 1.0));

        for (let d = 0; d < nc; d++) {
          const diff = embedding[i * nc + d] - embedding[kk * nc + d];
          forces[i * nc + d] += clip(gradCoeffRep * diff);
        }
      }
    }

    // Apply accumulated forces once per epoch.
    for (let idx = 0; idx < forces.length; idx++) {
      embedding[idx] += alpha * forces[idx];
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
  const nc = nComponents;
  const nEdges = graph.rows.length;

  const head = graph.rows; // new-point indices
  const tail = graph.cols; // training-point indices

  const epochOfNextSample = new Float32Array(epochsPerSample);

  // Forces accumulator over the new points only (training stays fixed).
  const forces = new Float32Array(nNew * nc);

  for (let epoch = 0; epoch < nEpochs; epoch++) {
    onProgress?.(epoch, nEpochs);
    const alpha = 1.0 - epoch / nEpochs;
    forces.fill(0);

    for (let edgeIdx = 0; edgeIdx < nEdges; edgeIdx++) {
      if (epochOfNextSample[edgeIdx] > epoch) continue;

      const i = head[edgeIdx]; // new point
      const j = tail[edgeIdx]; // training neighbor

      // Attraction: pull new point toward fixed training neighbor.
      let distSq = 0;
      for (let d = 0; d < nc; d++) {
        const diff = embeddingNew[i * nc + d] - embeddingTrain[j * nc + d];
        distSq += diff * diff;
      }
      distSq = Math.max(distSq, 1e-6);

      const powB = Math.pow(distSq, b);
      const gradCoeffAttr = (-2.0 * a * b * (powB / distSq)) / (a * powB + 1.0);

      for (let d = 0; d < nc; d++) {
        const diff = embeddingNew[i * nc + d] - embeddingTrain[j * nc + d];
        forces[i * nc + d] += clip(gradCoeffAttr * diff);
      }

      epochOfNextSample[edgeIdx] += epochsPerSample[edgeIdx];

      // Repulsion: push the new point away from random training points.
      for (let s = 0; s < negativeSampleRate; s++) {
        const kk = Math.floor(rng() * nTrain);

        let negDistSq = 0;
        for (let d = 0; d < nc; d++) {
          const diff = embeddingNew[i * nc + d] - embeddingTrain[kk * nc + d];
          negDistSq += diff * diff;
        }
        negDistSq = Math.max(negDistSq, 1e-6);

        const negPowB = Math.pow(negDistSq, b);
        const gradCoeffRep =
          (2.0 * gamma * b) / ((0.001 + negDistSq) * (a * negPowB + 1.0));

        for (let d = 0; d < nc; d++) {
          const diff = embeddingNew[i * nc + d] - embeddingTrain[kk * nc + d];
          forces[i * nc + d] += clip(gradCoeffRep * diff);
        }
      }
    }

    for (let idx = 0; idx < forces.length; idx++) {
      embeddingNew[idx] += alpha * forces[idx];
    }
  }

  return embeddingNew;
}
