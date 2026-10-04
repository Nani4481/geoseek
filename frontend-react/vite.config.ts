import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
// The vendored, hash-pinned third-party files (three.js, OrbitControls, Leaflet, the Earth textures, the web fonts).
// They are only ever READ from here: the build bundles those exact bytes. Provenance: geoseek.staging.vendor_provenance.
const VENDOR = path.resolve(repo, 'src/geoseek/analyst/vendor');

export default defineConfig({
  base: '/react/',
  plugins: [react()],
  resolve: {
    alias: {
      three: path.join(VENDOR, 'three.module.min.js'),
      leaflet: path.join(VENDOR, 'leaflet/leaflet.js'),
      '@vendor': VENDOR,
      '@webfonts': path.join(VENDOR, 'fonts'),
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
