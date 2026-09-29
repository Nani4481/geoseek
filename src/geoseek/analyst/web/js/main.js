import { initShell, initKeyboardNav } from "./shell.js";
import { startRouter } from "./router.js";
import { runIntro } from "./intro.js";

import "./screens/candidate.js";
import "./screens/queue.js";
import "./screens/overview.js";
import "./screens/search.js";
import "./screens/detect.js";
import "./screens/discovery.js";
import "./screens/watch.js";

import { wireGlobalDecisionKeys } from "./screens/candidate.js";

async function boot() {
  await initShell();
  const { onConfirm, onReject } = wireGlobalDecisionKeys();
  initKeyboardNav({ onConfirm, onReject });
  startRouter();
}

runIntro();
boot();
