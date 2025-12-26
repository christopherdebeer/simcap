import { defineConfig, Plugin, loadEnv } from 'vite';
import { resolve } from 'path';
import { IncomingMessage, ServerResponse } from 'http';
import { config as dotenvConfig } from 'dotenv';

// Load .env.local for server-side API handlers
dotenvConfig({ path: '.env.local' });

// API handler plugin for development
function apiPlugin(): Plugin {
  return {
    name: 'api-handler',
    configureServer(server) {
      server.middlewares.use(async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
        if (!req.url?.startsWith('/api/')) {
          return next();
        }

        try {
          // Extract the API route path
          const urlPath = req.url.split('?')[0];
          const apiPath = urlPath.replace('/api/', '');
          const handlerPath = resolve(__dirname, `api/${apiPath}.ts`);

          // Use Vite's ssrLoadModule to properly handle TypeScript
          const handler = await server.ssrLoadModule(handlerPath);
          
          // Collect request body for POST/PUT
          let body = '';
          if (req.method === 'POST' || req.method === 'PUT') {
            body = await new Promise<string>((resolve) => {
              const chunks: Buffer[] = [];
              req.on('data', (chunk: Buffer) => chunks.push(chunk));
              req.on('end', () => resolve(Buffer.concat(chunks).toString()));
            });
          }

          // Create a mock Request object
          const url = new URL(req.url, `http://${req.headers.host}`);
          const mockRequest = new Request(url.toString(), {
            method: req.method,
            headers: req.headers as HeadersInit,
            body: body || undefined,
          });

          // Call the handler
          const response = await handler.default(mockRequest);

          // Send the response
          res.statusCode = response.status;
          response.headers.forEach((value: string, key: string) => {
            res.setHeader(key, value);
          });

          const responseBody = await response.text();
          res.end(responseBody);
        } catch (error: any) {
          console.error('API handler error:', error);
          res.statusCode = 500;
          res.setHeader('Content-Type', 'application/json');
          res.end(JSON.stringify({ error: error.message }));
        }
      });
    }
  };
}

export default defineConfig({
  root: '.',

  // Path aliases matching tsconfig
  resolve: {
    alias: {
      '@api': resolve(__dirname, 'packages/api/src'),
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
        // Apps directory structure
        gambit: resolve(__dirname, 'apps/gambit/index.html'),
        collector: resolve(__dirname, 'apps/gambit/collector.html'),
        synth: resolve(__dirname, 'apps/gambit/synth.html'),
        ffo: resolve(__dirname, 'apps/ffo/index.html'),
        viz: resolve(__dirname, 'apps/viz/index.html'),
        loader: resolve(__dirname, 'apps/loader/index.html'),
        // Documentation viewer module
        'docs-viewer': resolve(__dirname, 'src/docs/index.ts'),
      },
      output: {
        // Ensure docs-viewer.js is placed in docs/ directory
        entryFileNames: (chunkInfo) => {
          if (chunkInfo.name === 'docs-viewer') {
            return 'docs/docs-viewer.js';
          }
          return 'assets/[name]-[hash].js';
        }
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
  publicDir: 'public',

  // Plugins
  plugins: [apiPlugin()]
});
