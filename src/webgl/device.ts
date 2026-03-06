/**
 * WebGL availability detection.
 *
 * WebGL 2 (based on OpenGL ES 3.0) is supported in all modern browsers and
 * provides compute-like capabilities through transform feedback. We use it
 * as a middle-ground fallback between WebGPU and pure CPU.
 */

let cachedAvailable: boolean | null = null;

/**
 * Synchronous check: returns `true` if a WebGL 2 context can be created.
 *
 * The result is cached after the first call. Unlike `isWebGPUAvailable()`,
 * this is a reliable check — creating an OffscreenCanvas + getContext is
 * synchronous and deterministic.
 */
export function isWebGLAvailable(): boolean {
  if (cachedAvailable !== null) return cachedAvailable;

  try {
    if (typeof OffscreenCanvas !== 'undefined') {
      const canvas = new OffscreenCanvas(1, 1);
      const gl = canvas.getContext('webgl2');
      cachedAvailable = gl !== null;
      return cachedAvailable;
    }

    // Node.js / environments without OffscreenCanvas
    if (typeof document !== 'undefined') {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2');
      cachedAvailable = gl !== null;
      return cachedAvailable;
    }

    cachedAvailable = false;
    return false;
  } catch {
    cachedAvailable = false;
    return false;
  }
}

/** Reset the cached result (used in tests). */
export function resetWebGLCache(): void {
  cachedAvailable = null;
}
