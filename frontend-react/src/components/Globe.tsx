/* eslint-disable @typescript-eslint/no-explicit-any */
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from '@vendor/OrbitControls.js';
import dayUrl from '@vendor/earth/day.jpg';
import nightUrl from '@vendor/earth/night.jpg';

export interface Pin {
  id: string;
  lon: number;
  lat: number;
  label: string;
  color: string;
  /** pulsing ring + always-on label */
  emphasis?: boolean;
  size?: number;
}
/** `lift` pulls the camera out mid-flight (arc) before it dives back in; `ms` is the flight duration. */
export interface Focus { lon: number; lat: number; key: string | number; distance?: number; lift?: number; ms?: number }

// Equirectangular lon/lat -> unit sphere, matching the vendored texture set (row 0 = north, Greenwich at the seam).
function toVec(lat: number, lon: number, r: number) {
  const phi = ((90 - lat) * Math.PI) / 180, theta = ((lon + 180) * Math.PI) / 180;
  return new THREE.Vector3(-r * Math.sin(phi) * Math.cos(theta), r * Math.cos(phi), r * Math.sin(phi) * Math.sin(theta));
}

// Subsolar point from the real UTC date/time (declination from day of year, longitude from the clock).
function sunDirection(d: Date) {
  const start = Date.UTC(d.getUTCFullYear(), 0, 0);
  const doy = (d.getTime() - start) / 86400000;
  const decl = 23.44 * Math.sin((2 * Math.PI / 365) * (doy - 81));
  const h = d.getUTCHours() + d.getUTCMinutes() / 60 + d.getUTCSeconds() / 3600;
  let lon = -(h - 12) * 15;
  lon = ((((lon + 180) % 360) + 360) % 360) - 180;
  return toVec(decl, lon, 1).normalize();
}

const EARTH_VS = `
varying vec2 vUv; varying vec3 vWorldNormal;
void main() { vUv = uv; vWorldNormal = normalize(mat3(modelMatrix) * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const EARTH_FS = `
uniform sampler2D dayMap; uniform sampler2D nightMap; uniform vec3 sunDir;
varying vec2 vUv; varying vec3 vWorldNormal;
void main() {
  float d = dot(normalize(vWorldNormal), normalize(sunDir));
  float t = smoothstep(-0.12, 0.22, d);
  vec3 day = texture2D(dayMap, vUv).rgb * (0.50 + 0.62 * max(d, 0.0));
  vec3 night = texture2D(nightMap, vUv).rgb * 1.5 + vec3(0.012, 0.03, 0.08);
  vec3 col = mix(night, day, t) * vec3(0.88, 0.95, 1.06);
  gl_FragColor = vec4(col, 1.0);
}`;
const ATMO_VS = `varying vec3 vN; void main() { vN = normalize(normalMatrix * normal); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const ATMO_FS = `varying vec3 vN; void main() { float i = pow(0.74 - dot(vN, vec3(0.0, 0.0, 1.0)), 3.2); gl_FragColor = vec4(0.30, 0.62, 1.0, 1.0) * i * 1.25; }`;

interface Props {
  pins: Pin[];
  focus?: Focus | null;
  selectedPin?: string | null;
  onPinClick?: (id: string) => void;
  autoRotate?: boolean;
  /** 0 = whole planet, 1 = very close */
  className?: string;
}

export function Globe({ pins, focus, selectedPin, onPinClick, autoRotate = true, className = '' }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const labelRefs = useRef(new Map<string, HTMLDivElement>());
  const ctl = useRef<{ flyTo: (f: Focus) => void; setPins: (p: Pin[]) => void; setSelected: (id: string | null) => void } | null>(null);
  const clickRef = useRef(onPinClick);
  clickRef.current = onPinClick;
  const pinsRef = useRef(pins);
  pinsRef.current = pins;
  const [failed, setFailed] = useState<string | null>(null);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    let renderer: any;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch (e) {
      setFailed('WebGL is not available in this browser.');
      return;
    }
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    el.insertBefore(renderer.domElement, el.firstChild);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 200);
    camera.position.copy(toVec(22, 78, 3.4));

    const loader = new THREE.TextureLoader();
    const day = loader.load(dayUrl), night = loader.load(nightUrl);
    const earthMat = new THREE.ShaderMaterial({
      uniforms: { dayMap: { value: day }, nightMap: { value: night }, sunDir: { value: sunDirection(new Date()) } },
      vertexShader: EARTH_VS, fragmentShader: EARTH_FS,
    });
    const earth = new THREE.Mesh(new THREE.SphereGeometry(1, 96, 96), earthMat);
    scene.add(earth);
    const atmo = new THREE.Mesh(new THREE.SphereGeometry(1.14, 64, 64),
      new THREE.ShaderMaterial({ vertexShader: ATMO_VS, fragmentShader: ATMO_FS, side: THREE.BackSide, blending: THREE.AdditiveBlending, transparent: true, depthWrite: false }));
    scene.add(atmo);

    // starfield
    const N = 1500, pos = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) {
      const v = new THREE.Vector3().randomDirection().multiplyScalar(40 + Math.random() * 30);
      pos.set([v.x, v.y, v.z], i * 3);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({ color: 0x9db6ff, size: 0.12, sizeAttenuation: true, transparent: true, opacity: 0.75 }));
    scene.add(stars);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true; controls.dampingFactor = 0.08;
    controls.enablePan = false; controls.minDistance = 1.35; controls.maxDistance = 5;
    controls.rotateSpeed = 0.55; controls.zoomSpeed = 0.7;
    controls.autoRotate = autoRotate; controls.autoRotateSpeed = 0.35;
    let idleTimer = 0;
    controls.addEventListener('start', () => { controls.autoRotate = false; window.clearTimeout(idleTimer); });
    controls.addEventListener('end', () => { if (autoRotate) idleTimer = window.setTimeout(() => { controls.autoRotate = true; }, 6000); });

    // pins
    const pinGroup = new THREE.Group();
    scene.add(pinGroup);
    const pinObjs: { id: string; mesh: any; ring: any; normal: any; emphasis: boolean }[] = [];
    let selected: string | null = null;
    const buildPins = (list: Pin[]) => {
      for (const o of pinObjs) { o.mesh.geometry.dispose(); o.mesh.material.dispose(); o.ring.geometry.dispose(); o.ring.material.dispose(); }
      pinObjs.length = 0; pinGroup.clear();
      for (const p of list) {
        const n = toVec(p.lat, p.lon, 1).normalize();
        const s = (p.size ?? 1) * 0.016;
        const mesh = new THREE.Mesh(new THREE.SphereGeometry(s, 16, 16), new THREE.MeshBasicMaterial({ color: new THREE.Color(p.color) }));
        mesh.position.copy(n.clone().multiplyScalar(1.008));
        mesh.userData.id = p.id;
        const ring = new THREE.Mesh(new THREE.RingGeometry(s * 1.5, s * 1.9, 40),
          new THREE.MeshBasicMaterial({ color: new THREE.Color(p.color), transparent: true, opacity: 0.8, side: THREE.DoubleSide, depthWrite: false }));
        ring.position.copy(n.clone().multiplyScalar(1.009));
        ring.lookAt(n.clone().multiplyScalar(2));
        pinGroup.add(mesh, ring);
        pinObjs.push({ id: p.id, mesh, ring, normal: n, emphasis: !!p.emphasis });
      }
    };
    buildPins(pinsRef.current);

    // camera fly-to
    let fly: { from: any; to: any; t0: number; dur: number; lift: number } | null = null;
    const flyTo = (f: Focus) => {
      controls.autoRotate = false;
      const to = toVec(f.lat, f.lon, f.distance ?? 2.1);
      fly = { from: camera.position.clone(), to, t0: performance.now(), dur: f.ms ?? 1500, lift: f.lift ?? 0 };
    };

    // hover + click
    const ray = new THREE.Raycaster();
    ray.params.Points = { threshold: 0.02 };
    const mouse = new THREE.Vector2();
    let hover: string | null = null, down = { x: 0, y: 0 };
    const hit = (e: PointerEvent) => {
      const r = renderer.domElement.getBoundingClientRect();
      mouse.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(mouse, camera);
      // enlarge hit area by testing slightly bigger invisible radius via distance-to-ray
      let best: string | null = null, bestD = 0.045;
      for (const o of pinObjs) {
        if (o.normal.dot(camera.position.clone().normalize()) < 0.15) continue;
        const d = ray.ray.distanceToPoint(o.mesh.position);
        if (d < bestD) { bestD = d; best = o.id; }
      }
      return best;
    };
    const onMove = (e: PointerEvent) => { hover = hit(e); renderer.domElement.style.cursor = hover ? 'pointer' : 'grab'; };
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; };
    const onUp = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > 5) return;
      const id = hit(e);
      if (id) clickRef.current?.(id);
    };
    renderer.domElement.addEventListener('pointermove', onMove);
    renderer.domElement.addEventListener('pointerdown', onDown);
    renderer.domElement.addEventListener('pointerup', onUp);

    const resize = () => {
      const w = el.clientWidth, h = el.clientHeight;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%';
      camera.aspect = w / h; camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(resize);
    ro.observe(el); resize();

    const tmp = new THREE.Vector3();
    let raf = 0, visible = true;
    const io = new IntersectionObserver((es) => { visible = es[0]?.isIntersecting ?? true; });
    io.observe(el);
    const loop = () => {
      raf = requestAnimationFrame(loop);
      if (!visible) return;
      const now = performance.now();
      if (fly) {
        const k = Math.min(1, (now - fly.t0) / fly.dur), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
        const from = fly.from.clone().normalize(), to = fly.to.clone().normalize();
        const q = new THREE.Quaternion().setFromUnitVectors(from, to);
        const qi = new THREE.Quaternion().identity().slerp(q, e);
        const dir = from.clone().applyQuaternion(qi);
        const dist = fly.from.length() + (fly.to.length() - fly.from.length()) * e + fly.lift * Math.sin(Math.PI * e);
        camera.position.copy(dir.multiplyScalar(dist));
        if (k >= 1) fly = null;
      }
      controls.update();
      // ring pulse + facing
      const camDir = camera.position.clone().normalize();
      const t = now / 1000;
      const k = camera.position.length() / 3.4; // keep pins a constant on-screen size as the camera zooms
      for (const o of pinObjs) {
        const facing = o.normal.dot(camDir) > 0.02;
        o.mesh.visible = o.ring.visible = facing;
        const pulse = o.emphasis || o.id === selected ? 1 + 0.55 * ((t * 0.9) % 1) : 1;
        o.mesh.scale.setScalar(k);
        o.ring.scale.setScalar(pulse * k);
        o.ring.material.opacity = o.emphasis || o.id === selected ? 0.85 * (1 - ((t * 0.9) % 1)) + 0.1 : 0.45;
      }
      // HTML labels
      for (const [id, div] of labelRefs.current) {
        const o = pinObjs.find((x) => x.id === id);
        if (!o) continue;
        const p = pinsRef.current.find((x) => x.id === id);
        const show = !!o.mesh.visible && (hover === id || selected === id || !!p?.emphasis);
        div.style.display = show ? 'block' : 'none';
        if (!show) continue;
        tmp.copy(o.mesh.position).project(camera);
        const w = el.clientWidth, h = el.clientHeight;
        div.style.transform = `translate(${(tmp.x * 0.5 + 0.5) * w + 10}px, ${(-tmp.y * 0.5 + 0.5) * h - 10}px)`;
      }
      renderer.render(scene, camera);
    };
    loop();

    ctl.current = {
      flyTo,
      setPins: buildPins,
      setSelected: (id) => { selected = id; },
    };
    return () => {
      cancelAnimationFrame(raf); window.clearTimeout(idleTimer); ro.disconnect(); io.disconnect();
      renderer.domElement.removeEventListener('pointermove', onMove);
      renderer.domElement.removeEventListener('pointerdown', onDown);
      renderer.domElement.removeEventListener('pointerup', onUp);
      controls.dispose();
      for (const o of pinObjs) { o.mesh.geometry.dispose(); o.mesh.material.dispose(); o.ring.geometry.dispose(); o.ring.material.dispose(); }
      day.dispose(); night.dispose(); earth.geometry.dispose(); earthMat.dispose(); atmo.geometry.dispose(); atmo.material.dispose();
      starGeo.dispose(); stars.material.dispose();
      renderer.dispose(); renderer.forceContextLoss?.();
      renderer.domElement.remove();
      ctl.current = null;
    };
    // the scene is built once; pins/focus/selection are pushed in through the handle below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { ctl.current?.setPins(pins); }, [pins]);
  useEffect(() => { ctl.current?.setSelected(selectedPin ?? null); }, [selectedPin]);
  useEffect(() => { if (focus) ctl.current?.flyTo(focus); }, [focus?.key]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div ref={wrap} className={`globe-wrap ${className}`}>
      {failed && <div className="empty" style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center' }}>3D globe unavailable — {failed}</div>}
      {pins.map((p) => (
        <div key={p.id} className="globe-label" style={{ display: 'none', borderColor: p.color }}
          ref={(d) => { if (d) labelRefs.current.set(p.id, d); else labelRefs.current.delete(p.id); }}>{p.label}</div>
      ))}
    </div>
  );
}
