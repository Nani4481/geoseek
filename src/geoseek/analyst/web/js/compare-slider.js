// compare-slider.js - the drag-to-compare surface shared by Candidate Detail
// (before/after imagery) and Object Detection (imagery vs. annotation
// layer). Mouse, touch and ArrowLeft/ArrowRight all move the same divider.
//
// Markup contract, built by mount():
//   <div class="compare-surface">
//     <div class="compare-base">...base layer (image / imagery)...</div>
//     <div class="compare-after-wrap"><div class="compare-after">...second layer...</div></div>
//     <div class="compare-handle" tabindex="0" role="slider" aria-label="...">
//       <div class="compare-handle-line"></div>
//       <div class="compare-handle-knob">↔</div>
//     </div>
//   </div>

export function mountCompareSlider(root, { initial = 50, ariaLabel = "Compare", onChange } = {}) {
  const handle = root.querySelector(".compare-handle");
  const afterWrap = root.querySelector(".compare-after-wrap");
  let pos = initial;
  let dragging = false;

  handle.setAttribute("role", "slider");
  handle.setAttribute("tabindex", "0");
  handle.setAttribute("aria-label", ariaLabel);
  handle.setAttribute("aria-valuemin", "2");
  handle.setAttribute("aria-valuemax", "98");

  function apply() {
    handle.style.left = pos + "%";
    afterWrap.style.clipPath = `inset(0 ${100 - pos}% 0 0)`;
    handle.setAttribute("aria-valuenow", Math.round(pos));
    if (onChange) onChange(pos);
  }

  function setFromClientX(clientX) {
    const rect = root.getBoundingClientRect();
    const pct = ((clientX - rect.left) / rect.width) * 100;
    pos = Math.min(98, Math.max(2, pct));
    apply();
  }

  function onMouseDown(e) {
    e.preventDefault();
    dragging = true;
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
  }
  function onMouseMove(e) { if (dragging) setFromClientX(e.clientX); }
  function onMouseUp() {
    dragging = false;
    window.removeEventListener("mousemove", onMouseMove);
    window.removeEventListener("mouseup", onMouseUp);
  }

  function onTouchStart(e) { dragging = true; e.preventDefault(); }
  function onTouchMove(e) {
    if (!dragging || !e.touches.length) return;
    setFromClientX(e.touches[0].clientX);
    e.preventDefault();
  }
  function onTouchEnd() { dragging = false; }

  function onKeyDown(e) {
    if (e.key === "ArrowLeft") { pos = Math.max(2, pos - 2); apply(); e.preventDefault(); }
    else if (e.key === "ArrowRight") { pos = Math.min(98, pos + 2); apply(); e.preventDefault(); }
    else if (e.key === "Home") { pos = 2; apply(); e.preventDefault(); }
    else if (e.key === "End") { pos = 98; apply(); e.preventDefault(); }
  }

  handle.addEventListener("mousedown", onMouseDown);
  handle.addEventListener("touchstart", onTouchStart, { passive: false });
  handle.addEventListener("touchmove", onTouchMove, { passive: false });
  handle.addEventListener("touchend", onTouchEnd);
  handle.addEventListener("keydown", onKeyDown);

  apply();

  return {
    setPosition(p) { pos = Math.min(98, Math.max(2, p)); apply(); },
    getPosition: () => pos,
    destroy() {
      handle.removeEventListener("mousedown", onMouseDown);
      handle.removeEventListener("touchstart", onTouchStart);
      handle.removeEventListener("touchmove", onTouchMove);
      handle.removeEventListener("touchend", onTouchEnd);
      handle.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    },
  };
}

export function compareSurfaceHtml({ baseHtml, afterHtml, cornerTL, cornerTR }) {
  return `
    <div class="compare-surface">
      <div class="compare-base">${baseHtml}</div>
      <div class="compare-after-wrap"><div class="compare-after">${afterHtml}</div></div>
      ${cornerTL ? `<div class="compare-corner-badge badge-plate tl">${cornerTL}</div>` : ""}
      ${cornerTR ? `<div class="compare-corner-badge badge-plate tr">${cornerTR}</div>` : ""}
      <div class="compare-handle">
        <div class="compare-handle-line"></div>
        <div class="compare-handle-knob">↔</div>
      </div>
    </div>`;
}
