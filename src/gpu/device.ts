/**
 * WebGPU device management — handles adapter/device acquisition and
 * provides a single shared device instance.
 */

let cachedDevice: GPUDevice | null = null;

/**
 * Request and cache a WebGPU device. Returns null if WebGPU is not available.
 */
export async function getGPUDevice(): Promise<GPUDevice | null> {
  if (cachedDevice) return cachedDevice;

  if (typeof navigator === 'undefined' || !navigator.gpu) {
    return null;
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return null;

  cachedDevice = await adapter.requestDevice();

  // Clear cache if the device is lost
  cachedDevice.lost.then(() => {
    cachedDevice = null;
  });

  return cachedDevice;
}

/**
 * Fast synchronous heuristic: returns `true` if `navigator.gpu` exists.
 *
 * **Caveat (Bug 13):** `navigator.gpu` being truthy does NOT guarantee that a
 * WebGPU adapter can be acquired — `requestAdapter()` may still return `null`
 * (no compatible hardware, or the browser has disabled WebGPU for the page).
 * Use `checkWebGPUAvailable()` for a reliable async check, or rely on the
 * `try/catch` around `GPUSgd.init()` in the calling code.
 */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/**
 * Reliably check whether WebGPU is usable in the current environment by
 * attempting to acquire an adapter via `getGPUDevice()`.
 *
 * Unlike the synchronous `isWebGPUAvailable()`, this actually calls
 * `navigator.gpu.requestAdapter()` and returns `false` if the adapter is
 * unavailable (no compatible GPU, browser policy, etc.).
 *
 * The result is automatically cached — repeated calls are free.
 */
export async function checkWebGPUAvailable(): Promise<boolean> {
  return (await getGPUDevice()) !== null;
}
