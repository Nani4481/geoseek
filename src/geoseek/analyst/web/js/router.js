// router.js - minimal hash router. Each screen registers a mount/show
// callback; the router shows exactly one <section class="screen"> at a
// time and never removes the others from the DOM.

const SCREENS = ["overview", "candidate", "queue", "search", "detect", "discovery", "watch"];

const registry = new Map(); // name -> { mount, show, mounted }
let current = null;

export function registerScreen(name, { mount, show }) {
  registry.set(name, { mount, show, mounted: false });
}

function parseHash() {
  const h = (location.hash || "#/overview").slice(2); // strip "#/"
  const [name, ...rest] = h.split("/");
  return { name: SCREENS.includes(name) ? name : "overview", param: rest.join("/") };
}

export function go(name, param) {
  location.hash = "#/" + name + (param ? "/" + param : "");
}

async function route() {
  const { name, param } = parseHash();
  document.querySelectorAll(".screen").forEach((el) => el.classList.remove("active"));
  document.querySelectorAll(".rail-btn").forEach((el) =>
    el.classList.toggle("active", el.dataset.screen === name));

  const entry = registry.get(name);
  if (!entry) return;
  const section = document.getElementById(`screen-${name}`);
  if (section) section.classList.add("active");

  if (!entry.mounted) {
    entry.mount();
    entry.mounted = true;
  }
  current = name;
  try {
    await entry.show(param);
  } catch (e) {
    console.error(`[router] show(${name}) failed`, e);
  }
}

export function currentScreen() { return current; }

export function startRouter() {
  window.addEventListener("hashchange", route);
  route();
}
