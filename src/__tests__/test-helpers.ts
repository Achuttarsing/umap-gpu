/** Compute per-edge epoch sampling periods (mirrors computeEpochsPerSample). */
export function computeEps(vals: Float32Array, nEpochs: number): Float32Array {
  const max = Math.max(...vals);
  const result = new Float32Array(vals.length);
  for (let i = 0; i < vals.length; i++) {
    const norm = vals[i] / max;
    result[i] = norm > 0 ? nEpochs / norm : -1;
  }
  return result;
}
