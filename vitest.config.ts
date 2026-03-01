import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      'hnswlib-wasm': path.resolve(
        __dirname,
        'src/__tests__/__stubs__/hnswlib-wasm.ts'
      ),
    },
  },
  test: {
    include: ['src/**/*.test.ts'],
    setupFiles: ['src/__tests__/setup-webgpu.ts'],
    testTimeout: 60_000,   // GPU init + 500 epochs on software renderer can be slow
    hookTimeout: 60_000,
  },
});
