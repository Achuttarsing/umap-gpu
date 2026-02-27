/// <reference types="@webgpu/types" />

// Vite raw import for the WGSL shader source
import shaderCode from './shaders/sgd.wgsl?raw';

export interface SGDParams {
  a: number;
  b: number;
  gamma: number;
  negativeSampleRate: number;
}

/**
 * GPU-accelerated SGD optimizer for UMAP embedding.
 * Each GPU thread processes one graph edge, applying attraction and repulsion forces.
 */
export class GPUSgd {
  private device!: GPUDevice;
  private pipeline!: GPUComputePipeline;

  async init(): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU not supported');
    this.device = await adapter.requestDevice();
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: shaderCode }),
        entryPoint: 'main',
      },
    });
  }

  /**
   * Run SGD optimization on the GPU.
   *
   * @param embedding   - Initial embedding positions [nVertices * nComponents]
   * @param head        - Edge source node indices
   * @param tail        - Edge target node indices
   * @param epochsPerSample - Per-edge epoch sampling period
   * @param nVertices   - Number of data points
   * @param nComponents - Embedding dimensionality (typically 2)
   * @param nEpochs     - Total number of optimization epochs
   * @param params      - UMAP curve parameters and repulsion settings
   * @returns Optimized embedding as Float32Array
   */
  async optimize(
    embedding: Float32Array,
    head: Uint32Array,
    tail: Uint32Array,
    epochsPerSample: Float32Array,
    nVertices: number,
    nComponents: number,
    nEpochs: number,
    params: SGDParams
  ): Promise<Float32Array> {
    const { device } = this;
    const nEdges = head.length;

    // Create GPU buffers
    const embeddingBuf = this.makeBuffer(
      embedding,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const headBuf = this.makeBuffer(head, GPUBufferUsage.STORAGE);
    const tailBuf = this.makeBuffer(tail, GPUBufferUsage.STORAGE);
    const epsBuf = this.makeBuffer(epochsPerSample, GPUBufferUsage.STORAGE);

    const epochNext = new Float32Array(nEdges).fill(0);
    const epochNextBuf = this.makeBuffer(epochNext, GPUBufferUsage.STORAGE);

    const epochNextNeg = new Float32Array(nEdges);
    for (let i = 0; i < nEdges; i++) {
      epochNextNeg[i] = epochsPerSample[i] / params.negativeSampleRate;
    }
    const epochNextNegBuf = this.makeBuffer(epochNextNeg, GPUBufferUsage.STORAGE);

    const seeds = new Uint32Array(nEdges);
    for (let i = 0; i < nEdges; i++) {
      seeds[i] = (Math.random() * 0xffffffff) | 0;
    }
    const seedsBuf = this.makeBuffer(seeds, GPUBufferUsage.STORAGE);

    // Params uniform buffer: 10 x 4 bytes = 40 bytes
    const paramsBuf = device.createBuffer({
      size: 40,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Run epochs
    for (let epoch = 0; epoch < nEpochs; epoch++) {
      const alpha = 1.0 - epoch / nEpochs; // linear LR decay

      // Write params uniform: 5 u32 + 4 f32 + 1 u32 = 40 bytes
      const paramsData = new ArrayBuffer(40);
      const u32View = new Uint32Array(paramsData);
      const f32View = new Float32Array(paramsData);

      u32View[0] = nEdges;
      u32View[1] = nVertices;
      u32View[2] = nComponents;
      u32View[3] = epoch;
      u32View[4] = nEpochs;
      f32View[5] = alpha;
      f32View[6] = params.a;
      f32View[7] = params.b;
      f32View[8] = params.gamma;
      u32View[9] = params.negativeSampleRate;

      device.queue.writeBuffer(paramsBuf, 0, paramsData);

      const bindGroup = device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: epsBuf } },
          { binding: 1, resource: { buffer: headBuf } },
          { binding: 2, resource: { buffer: tailBuf } },
          { binding: 3, resource: { buffer: embeddingBuf } },
          { binding: 4, resource: { buffer: epochNextBuf } },
          { binding: 5, resource: { buffer: epochNextNegBuf } },
          { binding: 6, resource: { buffer: paramsBuf } },
          { binding: 7, resource: { buffer: seedsBuf } },
        ],
      });

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(nEdges / 256));
      pass.end();
      device.queue.submit([encoder.finish()]);

      // Await GPU every 10 epochs to avoid TDR (GPU timeout)
      if (epoch % 10 === 0) {
        await device.queue.onSubmittedWorkDone();
      }
    }

    // Read back result
    const readBuf = device.createBuffer({
      size: embedding.byteLength,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(embeddingBuf, 0, readBuf, 0, embedding.byteLength);
    device.queue.submit([enc.finish()]);

    await readBuf.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(readBuf.getMappedRange().slice(0));
    readBuf.unmap();

    // Clean up buffers
    embeddingBuf.destroy();
    headBuf.destroy();
    tailBuf.destroy();
    epsBuf.destroy();
    epochNextBuf.destroy();
    epochNextNegBuf.destroy();
    seedsBuf.destroy();
    paramsBuf.destroy();
    readBuf.destroy();

    return result;
  }

  private makeBuffer(data: Float32Array | Uint32Array, usage: number): GPUBuffer {
    const buf = this.device.createBuffer({
      size: data.byteLength,
      usage,
      mappedAtCreation: true,
    });
    if (data instanceof Float32Array) {
      new Float32Array(buf.getMappedRange()).set(data);
    } else {
      new Uint32Array(buf.getMappedRange()).set(data);
    }
    buf.unmap();
    return buf;
  }
}
