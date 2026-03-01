/** Compute per-edge epoch sampling periods (mirrors computeEpochsPerSample). */
export function computeEps(vals: Float32Array, nEpochs: number): Float32Array {
  const max = Math.max(...vals);
  const result = new Float32Array(vals.length);
  for (let i = 0; i < vals.length; i++) {
    const norm = vals[i] / max;
    // Bug 7 fix: must be 1/norm, not nEpochs/norm.
    // The production computeEpochsPerSample uses 1/norm so that max-weight
    // edges fire every epoch (period = 1).  Using nEpochs/norm makes every
    // edge fire only once per run, leaving the SGD optimizer nearly idle.
    result[i] = norm > 0 ? 1.0 / norm : -1;
  }
  return result;
}
