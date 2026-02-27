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
 * Check whether WebGPU is available in the current environment.
 */
export function isWebGPUAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}
