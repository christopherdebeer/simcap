import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  root: '.',

  // Path aliases matching tsconfig
  resolve: {
    alias: {
      '@core': resolve(__dirname, 'packages/core/src'),
      '@filters': resolve(__dirname, 'packages/filters/src'),
      '@orientation': resolve(__dirname, 'packages/orientation/src'),
      '@puck': resolve(__dirname, 'packages/puck/src'),
    }
  },

  build: {
    outDir: 'dist',
    rollupOptions: {
      input: {
        // Root landing page
        main: resolve(__dirname, 'index.html'),
        // Current structure (will migrate to apps/ later)
        gambit: resolve(__dirname, 'src/web/GAMBIT/index.html'),
        collector: resolve(__dirname, 'src/web/GAMBIT/collector.html'),
        viz: resolve(__dirname, 'src/web/VIZ/index.html'),
        loader: resolve(__dirname, 'src/web/loader/index.html'),
      }
    }
  },

  // Dev server
  server: {
    port: 3000,
    open: '/'
  },

  // Optimize external CDN dependencies
  optimizeDeps: {
    include: ['three']
  },

  // Handle static assets
  publicDir: 'public'
});
