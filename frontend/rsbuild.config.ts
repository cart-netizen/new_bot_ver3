import {defineConfig} from "@rsbuild/core";
import {pluginReact} from "@rsbuild/plugin-react";

export default defineConfig({
  plugins: [
    pluginReact({
      swcReactOptions: {
        runtime: 'automatic',
      },
    }),
  ],

  source: {
    entry: {
      index: './src/main.tsx',
    },
    alias: {
      '@': './src',
    },
  },

  server: {
    port: 3000,
    host: '0.0.0.0',
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
      },
    },
  },

  html: {
    template: './index.html',
  },

  output: {
    target: 'web',
    distPath: {
      root: 'dist',
    },
    sourceMap: {
      js: 'source-map',
      css: true,
    },
  },

  // Удалена секция tools.postcss - используется отдельный файл

  performance: {
    chunkSplit: {
      strategy: 'split-by-experience',
    },
  },

  dev: {
    assetPrefix: true,
  },
});