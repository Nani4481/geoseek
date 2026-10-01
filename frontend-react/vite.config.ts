import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
// The existing frontend's vendored, hash-pinned third-party files. They are only ever READ from here:
// the React build bundles those exact bytes, it never edits or depends on that frontend running.
const WEB = path.resolve(repo, 'src/geoseek/analyst/web');

export default defineConfig({
  base: '/react/',
  plugins: [react()],
  resolve: {
    alias: {
      three: path.join(WEB, 'vendor/three.module.min.js'),
      leaflet: path.join(WEB, 'vendor/leaflet/leaflet.js'),
      '@vendor': path.join(WEB, 'vendor'),
      '@webfonts': path.join(WEB, 'fonts'),
      '@': path.join(here, 'src'),
    },
  },
  build: {
    outDir: path.resolve(repo, 'src/geoseek/analyst/web_react'),
    emptyOutDir: true,
    sourcemap: false,
    assetsInlineLimit: 0, // never inline assets as data: URIs - every asset is a scannable file
    modulePreload: { polyfill: false },
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    fs: { allow: [repo] },
    proxy: Object.fromEntries(
      ['/search', '/tile', '/health', '/stats', '/regions', '/presentation', '/candidates', '/restricted-zones',
       '/audit', '/export', '/watch-areas', '/notifications', '/sector-brief', '/discovery', '/detect', '/ui']
        .map((p) => [p, 'http://127.0.0.1:8000']),
    ),
  },
});
