// The three.js build and OrbitControls are the vendored, hash-pinned files from the existing frontend.
// They ship no type declarations; the 3D code treats them as untyped.
declare module 'three';
declare module '@vendor/OrbitControls.js' {
  export const OrbitControls: any;
}
declare module '*.jpg' { const url: string; export default url; }
declare module '*.png' { const url: string; export default url; }
declare module '*.woff2' { const url: string; export default url; }
