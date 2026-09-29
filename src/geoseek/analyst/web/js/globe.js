// globe.js - a physically-lit 3D Earth for the Overview screen's "GEOINT globe"
// view. Renders NASA Blue Marble day/night/specular/cloud imagery (staged
// offline by scripts/stage_earth_textures.py) on a sphere with a shader that
// blends day and night texels across a soft terminator, adds an ocean
// specular highlight and a fresnel atmosphere glow, and rotates the sphere
// under a sun direction computed from the real current UTC date/time - so
// the day/night line on screen roughly matches where it really is on Earth
// right now. No CDN, no network call once staged - three.js and the
// textures ship in vendor/.
import * as THREE from "three";
import { OrbitControls } from "../vendor/OrbitControls.js";

const EARTH_RADIUS = 1;
const TEX_BASE = new URL("../vendor/earth/", import.meta.url);

// exported so js/intro.js can build the same day/night-lit Earth for the
// boot animation without duplicating the shader source.
export function earthTexUrl(name) {
  return new URL(name, TEX_BASE).href;
}
const texUrl = earthTexUrl;

// Classic equirectangular lat/lon -> unit-sphere mapping for this exact
// texture set (row 0 = north pole, Greenwich meridian at the texture seam).
function latLonToVector3(lat, lon, radius) {
  const phi = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

// Approximate subsolar point (where the sun is directly overhead right now):
// solar declination from day-of-year, subsolar longitude from UTC clock time.
// A couple of degrees off the true value (no equation-of-time correction) -
// plenty accurate for a decorative terminator line on a background globe.
function subsolarPoint(date) {
  const start = Date.UTC(date.getUTCFullYear(), 0, 0);
  const dayOfYear = (date.getTime() - start) / 86400000;
  const declDeg = 23.44 * Math.sin((2 * Math.PI / 365) * (dayOfYear - 81));
  const utcHours = date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  let lon = -(utcHours - 12) * 15;
  lon = (((lon + 180) % 360) + 360) % 360 - 180;
  return { lat: declDeg, lon };
}

export function starTexture() {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.85)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function buildStarfield() {
  const COUNT = 4000;
  const positions = new Float32Array(COUNT * 3);
  const sizes = new Float32Array(COUNT);
  for (let i = 0; i < COUNT; i++) {
    const r = 40 + Math.random() * 20;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);
    sizes[i] = 0.08 + Math.random() * 0.22;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("size", new THREE.BufferAttribute(sizes, 1));
  const mat = new THREE.PointsMaterial({
    map: starTexture(), size: 0.14, sizeAttenuation: true,
    transparent: true, depthWrite: false, opacity: 0.85,
  });
  return new THREE.Points(geo, mat);
}

export const EARTH_VERTEX_SHADER = `
  varying vec2 vUv;
  varying vec3 vWorldNormal;
  varying vec3 vWorldPosition;
  void main() {
    vUv = uv;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPosition = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

export const EARTH_FRAGMENT_SHADER = `
  uniform sampler2D dayTexture;
  uniform sampler2D nightTexture;
  uniform sampler2D specularTexture;
  uniform vec3 sunDirection;
  varying vec2 vUv;
  varying vec3 vWorldNormal;
  varying vec3 vWorldPosition;

  void main() {
    vec3 n = normalize(vWorldNormal);
    vec3 sun = normalize(sunDirection);
    float cosine = dot(n, sun);
    float dayMix = smoothstep(-0.15, 0.15, cosine);

    vec3 dayColor = texture2D(dayTexture, vUv).rgb;
    vec3 nightColor = texture2D(nightTexture, vUv).rgb * 1.5;
    vec3 color = mix(nightColor, dayColor, dayMix);

    float specMask = texture2D(specularTexture, vUv).r;
    vec3 viewDir = normalize(cameraPosition - vWorldPosition);
    vec3 halfDir = normalize(sun + viewDir);
    float specAngle = max(dot(n, halfDir), 0.0);
    float spec = pow(specAngle, 34.0) * specMask * max(cosine, 0.0);
    color += vec3(1.0, 0.98, 0.9) * spec * 0.7;

    float rim = pow(1.0 - max(dot(n, viewDir), 0.0), 2.5) * max(cosine, 0.0);
    color += vec3(0.35, 0.55, 1.0) * rim * 0.2;

    gl_FragColor = vec4(color, 1.0);
  }
`;

const ATMOSPHERE_VERTEX_SHADER = `
  varying vec3 vNormalView;
  varying vec3 vWorldNormal;
  void main() {
    vNormalView = normalize(normalMatrix * normal);
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const ATMOSPHERE_FRAGMENT_SHADER = `
  uniform vec3 sunDirection;
  varying vec3 vNormalView;
  varying vec3 vWorldNormal;
  void main() {
    float rim = pow(0.75 - max(dot(vNormalView, vec3(0.0, 0.0, 1.0)), 0.0), 2.0);
    float sunFactor = clamp(dot(normalize(vWorldNormal), normalize(sunDirection)) * 0.5 + 0.5, 0.12, 1.0);
    vec3 col = mix(vec3(0.05, 0.09, 0.18), vec3(0.35, 0.62, 1.0), sunFactor);
    gl_FragColor = vec4(col * rim, rim);
  }
`;

function markerSprite(color) {
  const size = 64;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(0,0,0,0)";
  ctx.fill();
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2 - 4);
  g.addColorStop(0, color);
  g.addColorStop(0.5, color);
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, size / 2 - 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.arc(size / 2, size / 2, 5, 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function createGlobe(container, { onMarkerClick } = {}) {
  const loader = new THREE.TextureLoader();
  const dayTexture = loader.load(texUrl("day.jpg"));
  dayTexture.colorSpace = THREE.SRGBColorSpace;
  const nightTexture = loader.load(texUrl("night.jpg"));
  nightTexture.colorSpace = THREE.SRGBColorSpace;
  const specularTexture = loader.load(texUrl("specular.jpg"));
  const cloudsTexture = loader.load(texUrl("clouds.png"));
  cloudsTexture.colorSpace = THREE.SRGBColorSpace;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
  camera.position.set(0, 0.55, 2.6);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const { lat: sunLat, lon: sunLon } = subsolarPoint(new Date());
  const sunDirection = latLonToVector3(sunLat, sunLon, 1).normalize();

  const sun = new THREE.DirectionalLight(0xfff4e0, 2.0);
  sun.position.copy(sunDirection.clone().multiplyScalar(10));
  scene.add(sun);
  scene.add(new THREE.AmbientLight(0x404850, 0.55));

  const earthGroup = new THREE.Group();
  scene.add(earthGroup);

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
  const earth = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS, 96, 96), earthMaterial);
  earthGroup.add(earth);

  const cloudsMaterial = new THREE.MeshPhongMaterial({
    map: cloudsTexture, transparent: true, opacity: 0.55, depthWrite: false,
  });
  const clouds = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS * 1.008, 96, 96), cloudsMaterial);
  earthGroup.add(clouds);

  const atmosphereMaterial = new THREE.ShaderMaterial({
    uniforms: { sunDirection: { value: sunDirection } },
    vertexShader: ATMOSPHERE_VERTEX_SHADER,
    fragmentShader: ATMOSPHERE_FRAGMENT_SHADER,
    side: THREE.BackSide,
    blending: THREE.AdditiveBlending,
    transparent: true,
    depthWrite: false,
  });
  const atmosphere = new THREE.Mesh(new THREE.SphereGeometry(EARTH_RADIUS * 1.12, 64, 64), atmosphereMaterial);
  earthGroup.add(atmosphere);

  const markersGroup = new THREE.Group();
  earthGroup.add(markersGroup);

  scene.add(buildStarfield());

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.minDistance = 1.5;
  controls.maxDistance = 6;
  controls.enablePan = false;
  controls.rotateSpeed = 0.55;

  const aoiTex = markerSprite("#F0A030");
  const findingTex = markerSprite("#5FC8E8");
  const siteTex = markerSprite("#6B7480");

  function setMarkers({ aoi, findings, otherSites }) {
    while (markersGroup.children.length) markersGroup.remove(markersGroup.children[0]);
    // Every other tracked AOI, dim and non-interactive, so the globe reads
    // as the whole monitored network - the active site (amber, below) still
    // reads as "you are here" against them.
    (otherSites || []).forEach((s) => {
      if (s.lon == null || s.lat == null) return;
      const pos = latLonToVector3(s.lat, s.lon, EARTH_RADIUS * 1.006);
      const spriteMat = new THREE.SpriteMaterial({ map: siteTex, transparent: true, opacity: 0.75 });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      sprite.scale.set(0.05, 0.05, 1);
      sprite.renderOrder = 0;
      markersGroup.add(sprite);
    });
    if (aoi && aoi.lat != null && aoi.lon != null) {
      const pos = latLonToVector3(aoi.lat, aoi.lon, EARTH_RADIUS * 1.012);
      const spriteMat = new THREE.SpriteMaterial({ map: aoiTex, transparent: true });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      sprite.scale.set(0.09, 0.09, 1);
      sprite.renderOrder = 2;
      markersGroup.add(sprite);
    }
    (findings || []).forEach((f) => {
      if (f.lon == null || f.lat == null) return;
      const pos = latLonToVector3(f.lat, f.lon, EARTH_RADIUS * 1.008);
      const spriteMat = new THREE.SpriteMaterial({ map: findingTex, transparent: true });
      const sprite = new THREE.Sprite(spriteMat);
      sprite.position.copy(pos);
      // Sized for an actual mouse to land on, not just for looks - the
      // sprite's soft glow reads much smaller than its true (clickable)
      // quad, so a "visually right" size here was nearly impossible to hit.
      sprite.scale.set(0.09, 0.09, 1);
      sprite.userData.baseScale = 0.09;
      sprite.renderOrder = 1;
      sprite.userData.findingId = f.id;
      markersGroup.add(sprite);
    });
  }

  const raycaster = new THREE.Raycaster();
  const pointerNdc = new THREE.Vector2();
  let downPos = null;
  let hovered = null;

  function pointerToNdc(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointerNdc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointerNdc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  }

  // Sprites don't occlude against the opaque earth on their own (raycasting
  // ignores what's actually visible), so a marker on the globe's far side
  // could otherwise register a hit through the planet. Filter those out by
  // comparing distance-from-camera against the earth's near-side surface
  // along the same ray.
  function clickableHitAt(ndc) {
    raycaster.setFromCamera(ndc, camera);
    const markerHit = raycaster.intersectObjects(markersGroup.children, false)[0];
    if (!markerHit) return null;
    const earthHit = raycaster.intersectObject(earth, false)[0];
    if (earthHit && earthHit.distance < markerHit.distance - 0.01) return null;
    return markerHit;
  }

  function onPointerDown(e) { downPos = { x: e.clientX, y: e.clientY }; }
  function onPointerUp(e) {
    if (!onMarkerClick || !downPos) return;
    const moved = Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y);
    downPos = null;
    if (moved > 6) return; // a drag-to-rotate gesture, not a click
    pointerToNdc(e);
    const hit = clickableHitAt(pointerNdc);
    if (hit && hit.object.userData.findingId) onMarkerClick(hit.object.userData.findingId);
  }
  function onPointerMove(e) {
    // Only while not dragging - OrbitControls already owns pointermove
    // during an active drag, and re-raycasting every frame of a drag would
    // just be wasted work.
    if (downPos) return;
    pointerToNdc(e);
    const hit = clickableHitAt(pointerNdc);
    const next = hit && hit.object.userData.findingId ? hit.object : null;
    if (next === hovered) return;
    if (hovered) hovered.scale.setScalar(hovered.userData.baseScale ?? 0.09);
    hovered = next;
    if (hovered) hovered.scale.setScalar((hovered.userData.baseScale ?? 0.09) * 1.5);
    renderer.domElement.style.cursor = hovered ? "pointer" : "grab";
  }
  renderer.domElement.style.cursor = "grab";
  renderer.domElement.addEventListener("pointerdown", onPointerDown);
  renderer.domElement.addEventListener("pointerup", onPointerUp);
  if (onMarkerClick) renderer.domElement.addEventListener("pointermove", onPointerMove);

  function focusOn(lat, lon) {
    if (lat == null || lon == null) return;
    const target = latLonToVector3(lat, lon, 1);
    const dist = camera.position.length();
    const desired = target.clone().normalize().multiplyScalar(dist);
    camera.position.copy(desired);
    controls.update();
  }

  function resize() {
    const w = container.clientWidth, h = container.clientHeight;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  }

  let disposed = false;
  let raf = null;
  function tick() {
    if (disposed) return;
    // The router keeps every screen's DOM alive and only toggles a CSS
    // class to hide inactive ones, so this loop would otherwise keep
    // spinning (and burning GPU) forever once the analyst clicks away to
    // another screen. Fall back to slow polling while hidden instead of
    // scheduling a real animation frame, and resume full-rate rendering
    // the moment the globe is visible again.
    if (container.offsetParent === null) {
      raf = setTimeout(tick, 500);
      return;
    }
    earthGroup.rotation.y += 0.00035;
    clouds.rotation.y += 0.00006;
    controls.update();
    renderer.render(scene, camera);
    raf = requestAnimationFrame(tick);
  }

  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();
  tick();

  function dispose() {
    disposed = true;
    if (raf) { cancelAnimationFrame(raf); clearTimeout(raf); }
    ro.disconnect();
    renderer.domElement.removeEventListener("pointerdown", onPointerDown);
    renderer.domElement.removeEventListener("pointerup", onPointerUp);
    renderer.domElement.removeEventListener("pointermove", onPointerMove);
    controls.dispose();
    [dayTexture, nightTexture, specularTexture, cloudsTexture, aoiTex, findingTex, siteTex].forEach((t) => t.dispose());
    [earth.geometry, clouds.geometry, atmosphere.geometry].forEach((g) => g.dispose());
    [earthMaterial, cloudsMaterial, atmosphereMaterial].forEach((m) => m.dispose());
    renderer.dispose();
    if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
  }

  return { setMarkers, focusOn, dispose, resize };
}
