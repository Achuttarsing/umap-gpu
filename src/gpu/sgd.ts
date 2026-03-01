/// <reference types="@webgpu/types" />

// Vite raw import for the WGSL shader sources
import shaderCode from './shaders/sgd.wgsl?raw';
import applyForcesShaderCode from './shaders/apply-forces.wgsl?raw';

export interface SGDParams {
  a: number;
  b: number;
  gamma: number;
  negativeSampleRate: number;
}

/**
 * GPU-accelerated SGD optimizer for UMAP embedding.
 *
 * Uses a two-pass design per epoch to eliminate write-write races on shared
 * vertex positions (Bug 2 fix):
 *   Pass 1 (sgd.wgsl):           Each thread accumulates its attraction and
 *                                  repulsion gradients into an atomic<i32>
 *                                  forces buffer — no direct embedding writes.
 *   Pass 2 (apply-forces.wgsl):  Each thread applies one element's accumulated
 *                                  force to the embedding and resets the
 *                                  accumulator to zero for the next epoch.
 *
 * Both passes are submitted in the same command encoder so WebGPU guarantees
 * sequential execution and storage-buffer visibility between them.
 */
export class GPUSgd {
  private device!: GPUDevice;
  private sgdPipeline!: GPUComputePipeline;
  private applyForcesPipeline!: GPUComputePipeline;

  async init(): Promise<void> {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU not supported');
    this.device = await adapter.requestDevice();
    this.sgdPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: shaderCode }),
        entryPoint: 'main',
      },
    });
    this.applyForcesPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: this.device.createShaderModule({ code: applyForcesShaderCode }),
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
    params: SGDParams,
    onProgress?: (epoch: number, nEpochs: number) => void
  ): Promise<Float32Array> {
    const { device } = this;
    const nEdges = head.length;
    const nEmbeddingElements = nVertices * nComponents;

    // --- SGD pass buffers ---
    const embeddingBuf = this.makeBuffer(
      embedding,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
    );
    const headBuf = this.makeBuffer(head, GPUBufferUsage.STORAGE);
    const tailBuf = this.makeBuffer(tail, GPUBufferUsage.STORAGE);
    const epsBuf = this.makeBuffer(epochsPerSample, GPUBufferUsage.STORAGE);

    // Bug 4 fix: initialize epoch_of_next_sample to epochsPerSample (not 0).
    // The Python reference sets epoch_of_next_sample = epochs_per_sample.copy(),
    // so no edge fires in epoch 0 — they each wait at least one sampling period.
    const epochNext = new Float32Array(epochsPerSample);
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

    // Forces buffer: atomic<i32> accumulator, zero-initialized.
    // Shared between the SGD pass (writes) and the apply-forces pass (reads+resets).
    const forcesBuf = device.createBuffer({
      size: nEmbeddingElements * 4,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Int32Array(forcesBuf.getMappedRange()).fill(0);
    forcesBuf.unmap();

    // SGD params uniform buffer: 5×u32 + 4×f32 + 1×u32 = 40 bytes
    const sgdParamsBuf = device.createBuffer({
      size: 40,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Apply-forces params uniform buffer: u32 + f32 = 8 bytes (padded to 16 for alignment)
    const applyParamsBuf = device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Pre-create static apply-forces bind group (embedding + forces + params)
    const applyBindGroup = device.createBindGroup({
      layout: this.applyForcesPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: embeddingBuf } },
        { binding: 1, resource: { buffer: forcesBuf } },
        { binding: 2, resource: { buffer: applyParamsBuf } },
      ],
    });

    // Run epochs
    for (let epoch = 0; epoch < nEpochs; epoch++) {
      const alpha = 1.0 - epoch / nEpochs;

      // Write SGD params uniform
      const sgdParamsData = new ArrayBuffer(40);
      const u32View = new Uint32Array(sgdParamsData);
      const f32View = new Float32Array(sgdParamsData);
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
      device.queue.writeBuffer(sgdParamsBuf, 0, sgdParamsData);

      // Write apply-forces params uniform
      const applyParamsData = new ArrayBuffer(16);
      const applyU32 = new Uint32Array(applyParamsData);
      const applyF32 = new Float32Array(applyParamsData);
      applyU32[0] = nEmbeddingElements;
      applyF32[1] = alpha;
      device.queue.writeBuffer(applyParamsBuf, 0, applyParamsData);

      const sgdBindGroup = device.createBindGroup({
        layout: this.sgdPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: epsBuf } },
          { binding: 1, resource: { buffer: headBuf } },
          { binding: 2, resource: { buffer: tailBuf } },
          { binding: 3, resource: { buffer: embeddingBuf } },
          { binding: 4, resource: { buffer: epochNextBuf } },
          { binding: 5, resource: { buffer: epochNextNegBuf } },
          { binding: 6, resource: { buffer: sgdParamsBuf } },
          { binding: 7, resource: { buffer: seedsBuf } },
          { binding: 8, resource: { buffer: forcesBuf } },
        ],
      });

      // Submit both passes in one encoder: WebGPU guarantees sequential execution
      // and storage-buffer visibility between compute passes within the same submit.
      const encoder = device.createCommandEncoder();

      const sgdPass = encoder.beginComputePass();
      sgdPass.setPipeline(this.sgdPipeline);
      sgdPass.setBindGroup(0, sgdBindGroup);
      sgdPass.dispatchWorkgroups(Math.ceil(nEdges / 256));
      sgdPass.end();

      const applyPass = encoder.beginComputePass();
      applyPass.setPipeline(this.applyForcesPipeline);
      applyPass.setBindGroup(0, applyBindGroup);
      applyPass.dispatchWorkgroups(Math.ceil(nEmbeddingElements / 256));
      applyPass.end();

      device.queue.submit([encoder.finish()]);

      // Await GPU every 10 epochs to avoid TDR (GPU timeout).
      if (epoch % 10 === 0) {
        await device.queue.onSubmittedWorkDone();
        onProgress?.(epoch, nEpochs);
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
    forcesBuf.destroy();
    sgdParamsBuf.destroy();
    applyParamsBuf.destroy();
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
