import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';
import path from 'path';
import fs from 'fs';

/**
 * Lightweight manual Cesium integration (no vite-plugin-cesium).
 * - Copies node_modules/cesium/Build/Cesium -> <outDir>/cesium during build.
 * - Serves the same directory under /cesium during dev.
 * - Defines CESIUM_BASE_URL so Cesium can locate workers/assets/widgets.
 */
function cesiumStatic(): Plugin {
  const source = path.resolve(__dirname, 'node_modules/cesium/Build/Cesium');
  const baseUrl = '/cesium/';

  function copyDir(src: string, dest: string) {
    fs.mkdirSync(dest, { recursive: true });
    for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
      const s = path.join(src, entry.name);
      const d = path.join(dest, entry.name);
      if (entry.isDirectory()) copyDir(s, d);
      else fs.copyFileSync(s, d);
    }
  }

  return {
    name: 'mineralvision-cesium-static',
    config() {
      return {
        define: {
          CESIUM_BASE_URL: JSON.stringify(baseUrl),
        },
      };
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url || !req.url.startsWith(baseUrl)) return next();
        const rel = decodeURIComponent(req.url.slice(baseUrl.length).split('?')[0]);
        const file = path.join(source, rel);
        if (!file.startsWith(source) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
          return next();
        }
        const ext = path.extname(file).toLowerCase();
        const mime: Record<string, string> = {
          '.js': 'application/javascript',
          '.css': 'text/css',
          '.json': 'application/json',
          '.wasm': 'application/wasm',
          '.png': 'image/png',
          '.jpg': 'image/jpeg',
          '.jpeg': 'image/jpeg',
          '.svg': 'image/svg+xml',
          '.gif': 'image/gif',
          '.webp': 'image/webp',
          '.woff': 'font/woff',
          '.woff2': 'font/woff2',
          '.ttf': 'font/ttf',
        };
        if (mime[ext]) res.setHeader('Content-Type', mime[ext]);
        fs.createReadStream(file).pipe(res);
      });
    },
    closeBundle() {
      const outDir = path.resolve(__dirname, 'dist/cesium');
      if (fs.existsSync(source)) {
        copyDir(source, outDir);
      } else {
        console.warn('[cesium-static] Cesium build assets not found at', source);
      }
    },
  };
}

export default defineConfig({
  plugins: [
    react(),
    cesiumStatic(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico', 'apple-touch-icon.png', 'masked-icon.svg'],
      manifest: {
        name: 'MineralVision',
        short_name: 'MineralVision',
        description: 'AI-powered exploration and response decision support',
        theme_color: '#1e40af',
        background_color: '#0f172a',
        display: 'standalone',
        orientation: 'any',
        scope: '/',
        start_url: '/',
        shortcuts: [
          {
            name: 'Oil Spill Operations',
            short_name: 'Oil Spill',
            description: 'Triage and review oil-spill evidence',
            url: '/oil-spill',
            icons: [{ src: 'pwa-192x192.png', sizes: '192x192' }],
          },
          {
            name: 'Capture Field Evidence',
            short_name: 'Capture',
            description: 'Open field evidence capture',
            url: '/oil-spill?capture=1',
            icons: [{ src: 'pwa-192x192.png', sizes: '192x192' }],
          },
        ],
        icons: [
          {
            src: 'pwa-192x192.png',
            sizes: '192x192',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png'
          },
          {
            src: 'pwa-512x512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable'
          }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
        // Incident evidence is authenticated and never cached by the service worker.
        // The oil-spill page only keeps operator-submitted compact mask evidence in its
        // explicit local queue until a secure API connection is available.
        runtimeCaching: []
      }
    })
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  },
  build: {
    // Cesium is a large dependency; raise the chunk warning limit for its bundle.
    chunkSizeWarningLimit: 6000,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true
      },
      // Geotoolkit/innovation routers are mounted at the server root.
      '/innovations': {
        target: 'http://localhost:8000',
        changeOrigin: true
      }
    }
  }
});
