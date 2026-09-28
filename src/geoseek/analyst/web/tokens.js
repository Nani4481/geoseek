/* tokens.js — the single colour-resolution bridge between tokens.css's custom
   properties and anything that cannot consume a CSS variable directly: the 2D
   canvas map (CoordMap, app.js) and three.js (Globe, globe.js).

   Before this file existed, the audit (docs/FRONTEND_AUDIT.md Section 3) found
   THREE independent, hand-copied change-type palettes — CSS tokens, app.js's
   CTYPE_COLOR, globe.js's CTYPE_RGB — and one of the six colours (water_loss)
   had already drifted between them. This file is the fix: canvas/WebGL code
   never hand-copies a colour again, it asks Tokens for the one the CSS already
   defines.

   Classic (non-module) script — loaded via <script src="tokens.js"> BEFORE
   <script src="app.js">, so window.Tokens exists before app.js's top-level
   code runs. globe.js is an ES module (dynamically imported later, only once
   Overview mounts) and reads the same window.Tokens global explicitly.

   RESOLUTION STRATEGY
   For each token name, an offscreen probe element's `color` is set to
   `var(--token-name)`, then getComputedStyle(probe).color is painted into a
   1x1 canvas and read back with getImageData. A canvas 2D `fillStyle` setter
   accepts the full CSS Color 4 grammar — hex, rgb(), oklch(), the output of
   color-mix(), named colours, anything a browser's CSS parser accepts — and
   always canonicalises whatever it's given to straight (non-premultiplied)
   sRGB bytes on readback.

   This is deliberately MORE robust than parsing getComputedStyle's returned
   string as text: which literal syntax getComputedStyle serialises a colour
   back to (always "rgb(...)", vs. echoing "oklch(...)" verbatim) is not
   guaranteed to be stable across browser engines/versions once a token is
   authored in oklch()/color-mix() — which is exactly what tokens.css does.
   Painting to a canvas sidesteps the question of string format entirely: the
   canvas is asked to resolve the colour, not this script. */
"use strict";

(function () {
  const CHANGE_TYPES = ["water_gain", "water_loss", "construction", "clearance", "road", "other"];
  const DETECT_CLASSES = ["small-vehicle", "large-vehicle", "ship", "plane", "helicopter", "storage-tank", "harbor", "bridge"];

  // every colour token tokens.css defines — the whole map is resolved once at
  // boot ("cache the whole resolved map at boot"), not lazily per lookup.
  const ALL_TOKEN_NAMES = [
    "surface-app", "surface-panel", "surface-raised", "surface-overlay",
    "border-hairline", "border-strong",
    "text-primary", "text-secondary", "text-tertiary", "text-disabled",
    "accent", "accent-hover", "accent-active", "accent-subtle", "accent-gold",
    "success", "warning", "danger", "info",
    ...CHANGE_TYPES.map((t) => "change-" + t),
    ...CHANGE_TYPES.map((t) => "change-" + t + "-fill"),
    ...DETECT_CLASSES.map((c) => "detect-" + c),
  ];

  const RESOLVED = new Map(); // token name -> {r,g,b,a}
  const TRIPLES = new Map();  // token name -> stable, cached {r,g,b} (same object every call -
                               // callers that reference-compare two colours, e.g. CoordMap's
                               // "are all items in this cluster the same colour?" check, need the
                               // same name to always hand back the identical object, not a fresh
                               // {r,g,b} literal per call)
  let probe = null, ctx = null;

  function ensureRig() {
    if (probe) return;
    probe = document.createElement("span");
    // off-screen but NOT display:none — display:none elements do not reliably
    // participate in the style cascade the same way on every engine; absolute
    // positioning off-canvas keeps this a normal, fully-styled element.
    probe.style.cssText = "position:absolute;left:-9999px;top:-9999px;width:1px;height:1px;pointer-events:none;";
    document.body.appendChild(probe);
    const canvas = document.createElement("canvas");
    canvas.width = 1; canvas.height = 1;
    ctx = canvas.getContext("2d", { willReadFrequently: true });
  }

  // Never a real resolved token (this palette has no near-zero-alpha RGB
  // (1,2,3) colour) - used purely as a "did the next assignment actually
  // take" marker, see resolveOne() below.
  const SENTINEL = "rgba(1,2,3,0.004)";

  function resolveOne(name) {
    // Existence check #1, and the one that actually matters: an undefined
    // custom property does NOT make `color: var(--x)` invalid in a way
    // you'd notice from the computed .color alone - verified empirically. It
    // makes the declaration fall back to whatever colour this element would
    // have INHERITED anyway, which is a perfectly plausible, non-zero,
    // non-transparent colour. A check that only ever inspected the resolved
    // paint colour (all the original version of this file did) can never
    // catch a typo'd or missing token name this way. getPropertyValue on the
    // custom property itself has no such ambiguity: it is the empty string
    // if and only if nothing in the cascade defines --name.
    const raw = getComputedStyle(document.documentElement).getPropertyValue(`--${name}`).trim();
    if (!raw) {
      throw new Error(`[tokens.js] token "--${name}" is not defined anywhere in the cascade (empty computed value). Check tokens.css defines --${name} on :root.`);
    }

    probe.style.color = `var(--${name})`;
    const serialized = getComputedStyle(probe).color;

    // Existence/parse check #2: Canvas2D silently IGNORES a fillStyle
    // assignment it can't parse and keeps the PREVIOUS value - it does not
    // throw (verified empirically: assigning a garbage string after a known
    // value leaves fillStyle completely unchanged, no exception raised). A
    // try/catch around the assignment, as this file used to have, therefore
    // never fires - it cannot, because nothing throws. Set a sentinel no
    // real token could ever resolve to immediately before the real
    // assignment, and treat "fillStyle is still the sentinel afterwards" as
    // the rejection signal instead.
    ctx.fillStyle = SENTINEL;
    const sentinelCanonical = ctx.fillStyle;
    ctx.fillStyle = serialized;
    if (ctx.fillStyle === sentinelCanonical) {
      throw new Error(`[tokens.js] token "--${name}" resolved to a colour canvas could not parse ("${serialized}"). Fix tokens.css.`);
    }

    ctx.clearRect(0, 0, 1, 1); // transparent backdrop, so an alpha<1 token reads back unblended
    ctx.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = ctx.getImageData(0, 0, 1, 1).data;
    // A defined-but-fully-transparent token (e.g. "--x: transparent") is
    // almost certainly a mistake for anything meant to paint a visible
    // marker/dot. Existence is already proven above, so this check is
    // narrower now than before: it no longer needs to, and can't reliably,
    // double as the existence check too.
    if (r === 0 && g === 0 && b === 0 && a === 0) {
      throw new Error(`[tokens.js] token "--${name}" is defined but resolves to fully transparent black. Check tokens.css.`);
    }
    return { r, g, b, a: +(a / 255).toFixed(4) };
  }

  function resolveAll() {
    ensureRig();
    RESOLVED.clear();
    TRIPLES.clear();
    for (const name of ALL_TOKEN_NAMES) {
      const t = resolveOne(name);
      RESOLVED.set(name, t);
      TRIPLES.set(name, { r: t.r, g: t.g, b: t.b });
    }
  }

  function get(name) {
    const v = RESOLVED.get(name);
    if (!v) {
      throw new Error(`[tokens.js] Tokens API called for unknown token "${name}". ` +
        `If this is a real new token, add it to ALL_TOKEN_NAMES in tokens.js and define it in tokens.css.`);
    }
    return v;
  }

  const Tokens = {
    /** "rgb(r,g,b)" — for a solid canvas/WebGL fill or stroke. */
    rgb(name) {
      const t = get(name);
      return `rgb(${t.r},${t.g},${t.b})`;
    },
    /** "rgba(r,g,b,alpha)" — same colour, caller-chosen alpha (replaces the old hexA()). */
    rgba(name, alpha) {
      const t = get(name);
      return `rgba(${t.r},${t.g},${t.b},${alpha})`;
    },
    /** {r,g,b} 0-255 ints — for three.js vertex colours (divide by 255) or any
     *  caller that needs to compose its own rgba() string with a per-frame alpha.
     *  Returns the SAME cached object for a given name every time (see TRIPLES
     *  above) — safe to reference-compare (`a === b`) to test "same token". */
    triple(name) {
      get(name); // throws if unknown, same as the other accessors
      return TRIPLES.get(name);
    },
    /** {r,g,b} for a candidate's change_type string. Falls back to
     *  "change-other" and warns (not throws) for a change_type the backend
     *  sends that no token maps to — that is bad DATA, not bad CONFIG, and
     *  the two deserve different severities: a broken token name at boot is a
     *  build-time mistake (throw); an unrecognised value in a live API
     *  response is a runtime data-shape surprise the UI should degrade
     *  through, not crash on. */
    changeType(type) {
      const name = "change-" + type;
      if (!RESOLVED.has(name)) {
        console.warn(`[tokens.js] unrecognised change_type "${type}" — falling back to "other".`);
        return this.triple("change-other");
      }
      return this.triple(name);
    },
    /** {r,g,b} for an Object Detection class string (e.g. "small-vehicle").
     *  Same reasoning as changeType(): an unrecognised class from a live
     *  /detect/model-info response is a data surprise, not a config bug, so
     *  it warns and falls back to a neutral grey rather than throwing. */
    detectClass(cls) {
      const name = "detect-" + cls;
      if (!RESOLVED.has(name)) {
        console.warn(`[tokens.js] unrecognised detection class "${cls}" — falling back to neutral grey.`);
        return this.triple("text-tertiary");
      }
      return this.triple(name);
    },
    /** Re-resolve every token — call after switching [data-theme] on <html>/<body>. */
    refresh() {
      resolveAll();
    },
  };

  resolveAll();
  window.Tokens = Tokens;
})();
