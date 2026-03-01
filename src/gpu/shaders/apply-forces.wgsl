// Apply-forces shader — second pass of the two-pass GPU SGD.
//
// After the SGD pass has atomically accumulated all gradients into the forces
// buffer, this shader applies each element's accumulated force to the
// embedding and resets the accumulator to zero for the next epoch.

@group(0) @binding(0) var<storage, read_write> embedding : array<f32>;
@group(0) @binding(1) var<storage, read_write> forces    : array<atomic<i32>>;
@group(0) @binding(2) var<uniform>             params    : ApplyParams;

struct ApplyParams {
  n_elements : u32,   // nVertices * nComponents
  alpha      : f32,   // current learning rate
}

// Must match FORCE_SCALE in sgd.wgsl
const FORCE_SCALE : f32 = 65536.0;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= params.n_elements) { return; }

  // atomicExchange atomically reads the accumulated force and resets it to 0.
  let raw = atomicExchange(&forces[idx], 0);
  embedding[idx] += params.alpha * f32(raw) / FORCE_SCALE;
}
