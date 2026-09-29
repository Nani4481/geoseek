// theme.js - light/dark theme toggle. All color values live as CSS custom
// properties in css/tokens.css; this module only flips the `data-theme`
// attribute on <html> and remembers the choice. The attribute is also set
// synchronously by an inline script in index.html's <head> (before this
// module even loads) so the correct theme paints on the very first frame
// instead of flashing dark-then-light.
const STORAGE_KEY = "geoseek-theme";

export function getTheme() {
  return document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark";
}

export function setTheme(theme) {
  if (theme === "light") {
    document.documentElement.setAttribute("data-theme", "light");
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  try { localStorage.setItem(STORAGE_KEY, theme); } catch (e) { /* private mode - theme just won't persist */ }
}

export function toggleTheme() {
  const next = getTheme() === "light" ? "dark" : "light";
  setTheme(next);
  return next;
}
