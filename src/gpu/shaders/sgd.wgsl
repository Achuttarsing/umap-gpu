// UMAP SGD compute shader — processes one graph edge per GPU thread.
// Computes attraction and repulsion forces and accumulates them atomically
// into a forces buffer. A separate apply-forces pass then updates embeddings,
// eliminating write-write races on shared vertex positions.

@group(0) @binding(0) var<storage, read>       epochs_per_sample : array<f32>;
@group(0) @binding(1) var<storage, read>       head              : array<u32>;  // edge source
@group(0) @binding(2) var<storage, read>       tail              : array<u32>;  // edge target
@group(0) @binding(3) var<storage, read>       embedding         : array<f32>;  // [n * nComponents], read-only
@group(0) @binding(4) var<storage, read_write> epoch_of_next_sample : array<f32>;
@group(0) @binding(5) var<storage, read_write> epoch_of_next_negative_sample : array<f32>;
@group(0) @binding(6) var<uniform>             params            : Params;
@group(0) @binding(7) var<storage, read>       rng_seeds         : array<u32>;  // per-edge seed
@group(0) @binding(8) var<storage, read_write> forces            : array<atomic<i32>>;  // [n * nComponents]

// Scale factor for quantizing f32 gradients into i32 for atomic accumulation.
// Gradients are clipped to [-4, 4]. With up to ~1000 edges sharing a vertex
// the max accumulated magnitude is ~4000, well within i32 range at this scale.
const FORCE_SCALE : f32 = 65536.0;

struct Params {
  n_edges        : u32,
  n_vertices     : u32,
  n_components   : u32,
  current_epoch  : u32,
  n_epochs       : u32,
  alpha          : f32,   // learning rate (applied by apply-forces pass)
  a              : f32,
  b              : f32,
  gamma          : f32,   // repulsion strength
  negative_sample_rate : u32,
}

fn clip(v: f32, lo: f32, hi: f32) -> f32 {
  return max(lo, min(hi, v));
}

// Simple xorshift RNG per thread
fn xorshift(seed: u32) -> u32 {
  var s = seed;
  s ^= s << 13u;
  s ^= s >> 17u;
  s ^= s << 5u;
  return s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let edge_idx = gid.x;
  if (edge_idx >= params.n_edges) { return; }

  // Only process this edge if its epoch has come
  if (epoch_of_next_sample[edge_idx] > f32(params.current_epoch)) { return; }

  let i = head[edge_idx];
  let j = tail[edge_idx];
  let nc = params.n_components;

  // --- Attraction ---
  var dist_sq : f32 = 0.0;
  for (var d = 0u; d < nc; d++) {
    let diff = embedding[i * nc + d] - embedding[j * nc + d];
    dist_sq += diff * diff;
  }

  let pow_b = pow(dist_sq, params.b);
  // Guard dist_sq == 0: b-1 is negative so pow(0, b-1) = +Inf.
  let grad_coeff_attr = select(
    -2.0 * params.a * params.b * (pow_b / dist_sq) / (params.a * pow_b + 1.0),
    0.0,
    dist_sq == 0.0
  );

  for (var d = 0u; d < nc; d++) {
    let diff = embedding[i * nc + d] - embedding[j * nc + d];
    let grad = clip(grad_coeff_attr * diff, -4.0, 4.0);
    // Accumulate atomically to avoid write-write races across threads.
    atomicAdd(&forces[i * nc + d],  i32(grad * FORCE_SCALE));
    atomicAdd(&forces[j * nc + d], -i32(grad * FORCE_SCALE));
  }

  epoch_of_next_sample[edge_idx] += epochs_per_sample[edge_idx];

  // --- Repulsion (negative samples) ---
  // Compute how many negative samples are overdue relative to current epoch,
  // matching the Python reference: n_neg = floor((n - next_neg) / eps_per_neg).
  let epoch_f      = f32(params.current_epoch);
  let epochs_per_neg = epochs_per_sample[edge_idx] / f32(params.negative_sample_rate);
  var n_neg = 0u;
  if (epochs_per_neg > 0.0 && epoch_f >= epoch_of_next_negative_sample[edge_idx]) {
    n_neg = u32((epoch_f - epoch_of_next_negative_sample[edge_idx]) / epochs_per_neg);
    epoch_of_next_negative_sample[edge_idx] += f32(n_neg) * epochs_per_neg;
  }

  var rng = xorshift(rng_seeds[edge_idx] + params.current_epoch * 6364136223u);

  for (var s = 0u; s < n_neg; s++) {
    rng = xorshift(rng);
    let k = rng % params.n_vertices;
    if (k == i) { continue; }

    var neg_dist_sq : f32 = 0.0;
    for (var d = 0u; d < nc; d++) {
      let diff = embedding[i * nc + d] - embedding[k * nc + d];
      neg_dist_sq += diff * diff;
    }

    let grad_coeff_rep = 2.0 * params.gamma * params.b
                         / ((0.001 + neg_dist_sq) * (params.a * pow(neg_dist_sq, params.b) + 1.0));

    for (var d = 0u; d < nc; d++) {
      let diff = embedding[i * nc + d] - embedding[k * nc + d];
      let grad = clip(grad_coeff_rep * diff, -4.0, 4.0);
      atomicAdd(&forces[i * nc + d], i32(grad * FORCE_SCALE));
    }
  }
}
