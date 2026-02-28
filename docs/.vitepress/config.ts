import { defineConfig } from 'vitepress';
import llmstxt from 'vitepress-plugin-llms';

export default defineConfig({
  title: 'umap-gpu',
  description: 'UMAP with WebGPU-accelerated SGD and HNSW approximate nearest neighbors.',
  vite: {
    plugins: [llmstxt()],
  },
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/guide/api' },
    ],
    sidebar: [
      {
        text: 'Guide',
        items: [
          { text: 'Getting Started', link: '/guide/getting-started' },
          { text: 'Configuration', link: '/guide/configuration' },
          { text: 'API Reference', link: '/guide/api' },
          { text: 'Browser Support', link: '/guide/browser-support' },
        ],
      },
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/Achuttarsing/umap-gpu' },
    ],
  },
});
