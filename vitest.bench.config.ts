import { defineConfig } from 'vitest/config';
import path from 'node:path';

export default defineConfig({
  resolve: {
    alias: {
      'hnswlib-wasm': path.resolve(__dirname, 'node_modules/hnswlib-wasm/dist/hnswlib.js'),
    },
  },
  test: {
    include: ['benchmark/index.ts'],
    setupFiles: ['src/__tests__/setup-webgpu.ts'],
    testTimeout: 600_000,
    hookTimeout: 60_000,
    reporters: ['verbose'],
  },
});
