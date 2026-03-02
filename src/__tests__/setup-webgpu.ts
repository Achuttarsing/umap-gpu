/**
 * Vitest setupFile: prepares the environment for GPU tests.
 *
 * - Installs WebGPU constants (GPUBufferUsage, GPUMapMode, …) into globalThis
 *   so that gpu/sgd.ts can use them without explicit imports.
 * - Checks whether a real GPU adapter is accessible and sets the boolean flag
 *   `globalThis.__webGPUAvailable`, which is read synchronously at module-load
 *   time by umap-output-gpu.test.ts for `describe.skipIf`.
 *
 * IMPORTANT: this file does NOT install `navigator.gpu` globally. Doing so
 * would cause existing tests (umap-class.test.ts) to take the GPU code path,
 * breaking their progress-callback assertions that assume per-epoch callbacks.
 * Instead, umap-output-gpu.test.ts installs navigator.gpu locally in its own
 * beforeAll so that only its workers see a GPU device.
 */

try {
  const { create, globals } = await import('webgpu');

  // Install GPUBufferUsage, GPUMapMode, GPUShaderStage, … into globalThis.
  // These are plain numeric constants and don't affect other tests.
  Object.assign(globalThis, globals);

  // Check adapter availability using the package directly — does not require
  // navigator.gpu to be installed.
  const gpu = create([]);
  const adapter = await gpu.requestAdapter();
  (globalThis as Record<string, unknown>).__webGPUAvailable = adapter !== null;
} catch {
  // Package not installed, or Dawn failed to initialise.
  (globalThis as Record<string, unknown>).__webGPUAvailable = false;
}
