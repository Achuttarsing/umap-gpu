/**
 * Centralized SGD backend interface.
 *
 * Provides a unified API for the three SGD implementations (WebGPU, WebGL, CPU)
 * and a factory that selects the best available backend at runtime.
 *
 * Fallback order: WebGPU → WebGL → CPU
 */

import type { FuzzyGraph } from './fuzzy-set';

// ─── Common types ────────────────────────────────────────────────────────────

export interface SGDParams {
  a: number;
  b: number;
  gamma: number;
  negativeSampleRate: number;
}

export type ProgressCallback = (epoch: number, nEpochs: number) => void;

/** Identifier for each backend, useful for logging / debugging. */
export type BackendType = 'webgpu' | 'webgl' | 'cpu';

// ─── Backend interface ───────────────────────────────────────────────────────

/**
 * A backend that can run the SGD optimization step of the UMAP pipeline.
 * All backends accept the same inputs and produce the same output
 * (a Float32Array embedding), ensuring the rest of the pipeline is
 * backend-agnostic.
 */
export interface SGDBackend {
  /** Which implementation this backend uses. */
  readonly type: BackendType;

  /**
   * Run SGD optimization and return the optimized embedding.
   *
   * @param embedding       - Initial embedding positions [nVertices * nComponents]
   * @param graph           - Sparse fuzzy graph (rows, cols, vals)
   * @param epochsPerSample - Per-edge epoch sampling period
   * @param nVertices       - Number of data points
   * @param nComponents     - Embedding dimensionality (typically 2)
   * @param nEpochs         - Total number of optimization epochs
   * @param params          - UMAP curve parameters and repulsion settings
   * @param onProgress      - Optional progress callback
   */
  optimize(
    embedding: Float32Array,
    graph: FuzzyGraph,
    epochsPerSample: Float32Array,
    nVertices: number,
    nComponents: number,
    nEpochs: number,
    params: SGDParams,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array>;
}

// ─── Backend implementations ─────────────────────────────────────────────────

import { isWebGPUAvailable } from './gpu/device';
import { GPUSgd } from './gpu/sgd';
import { isWebGLAvailable } from './webgl/device';
import { WebGLSgd } from './webgl/sgd';
import { cpuSgd } from './fallback/cpu-sgd';

// -- WebGPU backend --

class WebGPUBackend implements SGDBackend {
  readonly type = 'webgpu' as const;

  async optimize(
    embedding: Float32Array,
    graph: FuzzyGraph,
    epochsPerSample: Float32Array,
    nVertices: number,
    nComponents: number,
    nEpochs: number,
    params: SGDParams,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array> {
    const gpu = new GPUSgd();
    await gpu.init();
    return gpu.optimize(
      embedding,
      new Uint32Array(graph.rows),
      new Uint32Array(graph.cols),
      epochsPerSample,
      nVertices,
      nComponents,
      nEpochs,
      params,
      onProgress,
    );
  }
}

// -- WebGL backend --

class WebGLBackend implements SGDBackend {
  readonly type = 'webgl' as const;

  async optimize(
    embedding: Float32Array,
    graph: FuzzyGraph,
    epochsPerSample: Float32Array,
    nVertices: number,
    nComponents: number,
    nEpochs: number,
    params: SGDParams,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array> {
    const gl = new WebGLSgd();
    gl.init();
    return gl.optimize(
      embedding,
      graph,
      epochsPerSample,
      nVertices,
      nComponents,
      nEpochs,
      params,
      onProgress,
    );
  }
}

// -- CPU backend --

class CPUBackend implements SGDBackend {
  readonly type = 'cpu' as const;

  async optimize(
    embedding: Float32Array,
    graph: FuzzyGraph,
    epochsPerSample: Float32Array,
    nVertices: number,
    nComponents: number,
    nEpochs: number,
    params: SGDParams,
    onProgress?: ProgressCallback,
  ): Promise<Float32Array> {
    return cpuSgd(
      embedding,
      graph,
      epochsPerSample,
      nVertices,
      nComponents,
      nEpochs,
      { a: params.a, b: params.b, gamma: params.gamma, negativeSampleRate: params.negativeSampleRate },
      onProgress,
    );
  }
}

// ─── Backend factory ─────────────────────────────────────────────────────────

/**
 * Select the best available SGD backend using the fallback chain:
 *   WebGPU → WebGL → CPU
 *
 * The returned backend is ready to use — call `.optimize()` directly.
 */
export async function selectBackend(): Promise<SGDBackend> {
  // 1. Try WebGPU
  if (isWebGPUAvailable()) {
    try {
      const gpu = new GPUSgd();
      await gpu.init();
      return new WebGPUBackend();
    } catch {
      // WebGPU init failed — fall through
    }
  }

  // 2. Try WebGL
  if (isWebGLAvailable()) {
    return new WebGLBackend();
  }

  // 3. CPU always available
  return new CPUBackend();
}

/**
 * Get a backend by explicit type. Throws if the requested backend is
 * unavailable. Useful for testing or when the caller knows which backend
 * they want.
 */
export function getBackend(type: BackendType): SGDBackend {
  switch (type) {
    case 'webgpu':
      return new WebGPUBackend();
    case 'webgl':
      return new WebGLBackend();
    case 'cpu':
      return new CPUBackend();
  }
}
