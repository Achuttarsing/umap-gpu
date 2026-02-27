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
});
