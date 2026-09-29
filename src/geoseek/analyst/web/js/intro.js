// intro.js - plays the one-time boot sequence: a 3D Earth materializing out
// of converging data-flow particles, reusing globe.js's day/night-lit Earth
// shader and the same staged Blue Marble textures (so the intro previews
// the exact globe the Overview screen's "GEOINT GLOBE" view shows later).
// #intro itself is already painted from pure HTML/CSS before this module
// loads, so there's no blank-page flash while JS boots - this module only
// builds the WebGL scene, ticks the status line, and decides when to fade
// the overlay out.
import * as THREE from "three";
import { earthTexUrl, starTexture, EARTH_VERTEX_SHADER, EARTH_FRAGMENT_SHADER } from "./globe.js";

const STATUS_LINES = [
  "MOUNTING LOCAL ARCHIVE",
  "LOADING RETRIEVAL INDEX",
  "PREPARING WORKSPACE",
  "READY",
];
const MIN_DISPLAY_MS = 1850;
const STATUS_INTERVAL_MS = 340;
const PARTICLE_COUNT = 900;
const ASSEMBLE_SECONDS = 1.35;

export function runIntro() {
  const el = document.getElementById("intro");
  if (!el) return;

  const reducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  if (reducedMotion) {
    el.classList.add("intro-hide");
    return;
  }

  const statusEl = document.getElementById("intro-status");
  let i = 0;
  if (statusEl) statusEl.textContent = STATUS_LINES[0];
  const tick = setInterval(() => {
    i = Math.min(i + 1, STATUS_LINES.length - 1);
    if (statusEl) statusEl.textContent = STATUS_LINES[i];
    if (i === STATUS_LINES.length - 1) clearInterval(tick);
  }, STATUS_INTERVAL_MS);

  let scene = null;
  try {
    scene = buildScene(document.getElementById("intro-scene"));
  } catch (e) {
    // WebGL unavailable or a texture failed to load - the CSS-only title/
    // bar/status overlay still works fine without the 3D backdrop.
    console.warn("[intro] 3D scene unavailable, falling back to overlay only", e);
  }

  let done = false;
  function finish() {
    if (done) return;
    done = true;
    clearInterval(tick);
    el.classList.add("intro-hide");
    // Let the 420ms CSS fade finish before tearing down the WebGL context,
    // so the scene is still there (behind the fading overlay) rather than
    // popping to black a beat early.
    setTimeout(() => scene?.dispose(), 460);
  }

  el.addEventListener("click", finish, { once: true });
  document.addEventListener("keydown", finish, { once: true });
  setTimeout(finish, MIN_DISPLAY_MS);
}

function easeOutCubic(x) { return 1 - Math.pow(1 - x, 3); }
function easeOutBack(x) {
  const c1 = 1.70158, c3 = c1 + 1;
  return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
}

function buildScene(container) {
  if (!container) return null;

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 0.25, 6.4);
  camera.lookAt(0, 0, 0);

  const sunDirection = new THREE.Vector3(0.55, 0.35, 0.85).normalize();
  const sun = new THREE.DirectionalLight(0xfff4e0, 2.2);
  sun.position.copy(sunDirection.clone().multiplyScalar(8));
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0x404850, 0.6));

  const loader = new THREE.TextureLoader();
  const dayTexture = loader.load(earthTexUrl("day.jpg"));
  dayTexture.colorSpace = THREE.SRGBColorSpace;
  const nightTexture = loader.load(earthTexUrl("night.jpg"));
  nightTexture.colorSpace = THREE.SRGBColorSpace;
  const specularTexture = loader.load(earthTexUrl("specular.jpg"));

  const earthMaterial = new THREE.ShaderMaterial({
    uniforms: {
      dayTexture: { value: dayTexture },
      nightTexture: { value: nightTexture },
      specularTexture: { value: specularTexture },
      sunDirection: { value: sunDirection },
    },
    vertexShader: EARTH_VERTEX_SHADER,
    fragmentShader: EARTH_FRAGMENT_SHADER,
  });
  const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 48), earthMaterial);
  earth.scale.setScalar(0.001);
  scene.add(earth);

  // Data-flow particles: start scattered on an outer shell, converge to a
  // loose halo just above the globe's surface as it assembles.
  const startPos = new Float32Array(PARTICLE_COUNT * 3);
  const endPos = new Float32Array(PARTICLE_COUNT * 3);
  for (let p = 0; p < PARTICLE_COUNT; p++) {
    const r0 = 5.2 + Math.random() * 3.2;
    const theta0 = Math.random() * Math.PI * 2;
    const phi0 = Math.acos(2 * Math.random() - 1);
    startPos[p * 3] = r0 * Math.sin(phi0) * Math.cos(theta0);
    startPos[p * 3 + 1] = r0 * Math.sin(phi0) * Math.sin(theta0);
    startPos[p * 3 + 2] = r0 * Math.cos(phi0);

    const r1 = 1.14 + Math.random() * 0.34;
    const theta1 = theta0 + (Math.random() - 0.5) * 0.5;
    const phi1 = Math.min(Math.PI - 0.05, Math.max(0.05, phi0 + (Math.random() - 0.5) * 0.5));
    endPos[p * 3] = r1 * Math.sin(phi1) * Math.cos(theta1);
    endPos[p * 3 + 1] = r1 * Math.sin(phi1) * Math.sin(theta1);
    endPos[p * 3 + 2] = r1 * Math.cos(phi1);
  }
  const particleGeo = new THREE.BufferGeometry();
  particleGeo.setAttribute("position", new THREE.BufferAttribute(startPos.slice(), 3));
  const particleMat = new THREE.PointsMaterial({
    map: starTexture(),
    color: 0xf0a030,
    size: 0.1,
    sizeAttenuation: true,
    transparent: true,
    depthWrite: false,
    opacity: 0.9,
    blending: THREE.AdditiveBlending,
  });
  const particles = new THREE.Points(particleGeo, particleMat);
  scene.add(particles);

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  const clock = new THREE.Clock();
  let disposed = false;
  let raf = null;
  function frame() {
    if (disposed) return;
    const t = clock.getElapsedTime();
    const progress = Math.min(t / ASSEMBLE_SECONDS, 1);
    const eased = easeOutCubic(progress);

    const posAttr = particleGeo.attributes.position;
    for (let p = 0; p < PARTICLE_COUNT; p++) {
      posAttr.array[p * 3] = startPos[p * 3] + (endPos[p * 3] - startPos[p * 3]) * eased;
      posAttr.array[p * 3 + 1] = startPos[p * 3 + 1] + (endPos[p * 3 + 1] - startPos[p * 3 + 1]) * eased;
      posAttr.array[p * 3 + 2] = startPos[p * 3 + 2] + (endPos[p * 3 + 2] - startPos[p * 3 + 2]) * eased;
    }
    posAttr.needsUpdate = true;
    particleMat.opacity = 0.9 * (1 - eased * 0.55);

    const growth = Math.min(1, Math.max(0, (t - 0.2) / 1.0));
    earth.scale.setScalar(THREE.MathUtils.lerp(0.001, 1, easeOutBack(growth)));
    earth.rotation.y = t * 0.18;
    camera.position.z = THREE.MathUtils.lerp(6.4, 3.7, eased);

    renderer.render(scene, camera);
    raf = requestAnimationFrame(frame);
  }
  raf = requestAnimationFrame(frame);

  function dispose() {
    disposed = true;
    if (raf) cancelAnimationFrame(raf);
    ro.disconnect();
    particleGeo.dispose();
    particleMat.dispose();
    particleMat.map?.dispose();
    earth.geometry.dispose();
    earthMaterial.dispose();
    [dayTexture, nightTexture, specularTexture].forEach((t) => t.dispose());
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  }

  return { dispose };
}
