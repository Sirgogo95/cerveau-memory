// Dev-only UI annotation capture. Auto-injected before </body> by
// kernel.base_app.middleware.AnnotationScriptMiddleware under DEBUG, so no template
// includes it by hand. It lives in the project-wide static/ folder, not in an app:
// it is injected into every page of the project, whichever app renders it.
// Alt+click any element to open a debug popup
// at the click point, type a note, save; then hit the floating "Send" button to
// POST the batch to the Django store view. Same-origin, so no CORS.
//
// Inspect from devtools: the pending queue is exposed as `window.annotations`
// (type `annotations` in the Console), and each capture/send is logged there too.
(function () {
  const ENDPOINT = "/__annotations__/";
  const pending = [];
  window.annotations = pending; // inspect in the Console: `annotations`

  // Persist the unsent queue to localStorage so it survives reloads and page
  // switches (per-origin). Cleared as items are successfully sent.
  const STORAGE_KEY = "__annotations_pending";
  function persist() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(pending));
    } catch (e) {
      /* private mode / disabled storage: in-memory only */
    }
  }
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (Array.isArray(saved)) pending.push(...saved);
  } catch (e) {
    /* ignore malformed storage */
  }

  function cssPath(el) {
    if (el.id) return `#${el.id}`;
    const cls = [...(el.classList || [])].join(".");
    return `${el.tagName.toLowerCase()}${cls ? "." + cls : ""}`;
  }

  function nearestTpl(el) {
    let node = el;
    while (node && node.nodeType === 1) {
      if (node.dataset && node.dataset.tpl) return node.dataset.tpl;
      node = node.parentElement;
    }
    return "";
  }

  function flash(el) {
    const old = el.style.outline;
    el.style.outline = "2px solid #f59e0b";
    setTimeout(() => (el.style.outline = old), 600);
  }

  function capture(target, note) {
    const record = {
      note,
      selector: cssPath(target),
      text: (target.innerText || "").slice(0, 120),
      tpl: nearestTpl(target),
      url: location.pathname,
      ts: Date.now(),
    };
    pending.push(record);
    persist();
    console.log("[annotations] captured:", record);
    flash(target);
    updateButton();
  }

  // --- Custom debug popup ----------------------------------------------------
  let styleInjected = false;
  function ensureStyle() {
    if (styleInjected) return;
    styleInjected = true;
    const s = document.createElement("style");
    s.textContent =
      ".dbg-annot{position:fixed;z-index:100000;width:288px;box-sizing:border-box;background:#17171f;" +
      "color:#e5e7eb;border:1px solid #f59e0b;border-radius:10px;padding:11px;" +
      "box-shadow:0 12px 34px rgba(0,0,0,.55);font:13px/1.45 system-ui;" +
      "animation:dbgpop .12s ease-out;}" +
      "@keyframes dbgpop{from{opacity:0;transform:scale(.96)}to{opacity:1;transform:scale(1)}}" +
      ".dbg-annot .hdr{display:flex;align-items:center;gap:6px;font-size:10px;" +
      "font-weight:700;letter-spacing:.14em;color:#f59e0b;text-transform:uppercase;}" +
      ".dbg-annot .loc{font:11px ui-monospace,Menlo,Consolas,monospace;color:#8b8b9a;" +
      "margin:5px 0 8px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}" +
      ".dbg-annot textarea{width:100%;box-sizing:border-box;background:#0f0f16;" +
      "color:#e5e7eb;border:1px solid #33333f;border-radius:6px;padding:7px;" +
      "resize:vertical;min-height:58px;font:13px/1.45 system-ui;outline:none;}" +
      ".dbg-annot textarea:focus{border-color:#f59e0b;}" +
      ".dbg-annot .row{display:flex;align-items:center;justify-content:space-between;" +
      "gap:10px;margin-top:10px;}" +
      ".dbg-annot .hint{font-size:10px;color:#6b7280;flex:1;min-width:0;" +
      "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}" +
      ".dbg-annot .actions{display:flex;gap:6px;flex-shrink:0;}" +
      ".dbg-annot button{border:1px solid transparent;border-radius:6px;padding:6px 12px;" +
      "cursor:pointer;font:13px system-ui;white-space:nowrap;}" +
      ".dbg-annot button.save{background:#f59e0b;color:#111;font-weight:600;}" +
      ".dbg-annot button.save:hover{background:#fbbf24;}" +
      ".dbg-annot button.cancel{background:transparent;color:#9ca3af;border-color:#33333f;}" +
      ".dbg-annot button.cancel:hover{color:#e5e7eb;border-color:#4b4b5a;}";
    document.head.appendChild(s);
  }

  let popup = null;
  function onOutside(e) {
    if (popup && !popup.contains(e.target)) closePopup();
  }
  function closePopup() {
    if (popup) {
      if (popup._restore) popup._restore();
      popup.remove();
      popup = null;
    }
    document.removeEventListener("mousedown", onOutside, true);
  }

  function openPopup(target, x, y) {
    closePopup();
    ensureStyle();

    const prevOutline = target.style.outline;
    target.style.outline = "2px dashed #f59e0b";

    popup = document.createElement("div");
    popup.className = "dbg-annot";
    popup._restore = () => (target.style.outline = prevOutline);
    popup.innerHTML =
      '<div class="hdr">🐛 debug annotation</div>' +
      '<div class="loc"></div>' +
      '<textarea placeholder="What should change here?"></textarea>' +
      '<div class="row"><span class="hint">Enter to save · Esc to cancel</span>' +
      '<span class="actions"><button class="cancel">Cancel</button>' +
      '<button class="save">Save</button></span></div>';
    // set location via textContent so a weird selector can't inject markup
    popup.querySelector(".loc").textContent = nearestTpl(target) || cssPath(target);
    document.body.appendChild(popup);

    // position at the click point, clamped into the viewport
    const r = popup.getBoundingClientRect();
    popup.style.left = Math.max(8, Math.min(x, window.innerWidth - r.width - 8)) + "px";
    popup.style.top = Math.max(8, Math.min(y, window.innerHeight - r.height - 8)) + "px";

    const ta = popup.querySelector("textarea");
    ta.focus();

    function done(save) {
      const note = ta.value.trim();
      closePopup();
      if (save && note) capture(target, note);
    }
    popup.querySelector("button.save").onclick = () => done(true);
    popup.querySelector("button.cancel").onclick = () => done(false);
    ta.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        done(true);
      } else if (ev.key === "Escape") {
        ev.preventDefault();
        done(false);
      }
    });

    // close when clicking elsewhere (next tick so the opening click is ignored)
    setTimeout(() => document.addEventListener("mousedown", onOutside, true), 0);
  }

  document.addEventListener("click", (e) => {
    if (popup && popup.contains(e.target)) return; // interacting with the popup
    if (e.target === btn) return; // the Send button
    if (!e.altKey) return; // Alt+click to annotate; normal clicks untouched
    e.preventDefault();
    openPopup(e.target, e.clientX, e.clientY);
  });

  // --- Floating, draggable Send button --------------------------------------
  const btn = document.createElement("button");
  btn.style.cssText =
    "position:fixed;bottom:20px;right:20px;z-index:99999;padding:10px 15px;" +
    "background:#17171f;color:#e5e7eb;border:1px solid #f59e0b;border-radius:10px;" +
    "cursor:grab;font:13px system-ui;font-weight:600;letter-spacing:.02em;" +
    "box-shadow:0 12px 34px rgba(0,0,0,.55);touch-action:none;user-select:none;";
  let moved = false; // set by a drag so the click that ends it does not send

  function updateButton() {
    btn.textContent = `🐛 Send ${pending.length} annotation(s)`;
    btn.style.opacity = pending.length ? "1" : "0.6";
  }

  btn.onclick = async () => {
    if (moved) {
      moved = false;
      return;
    } // ignore the click that ends a drag
    if (!pending.length) return;
    const batch = pending.splice(0, pending.length);
    btn.disabled = true;
    btn.textContent = "Sending...";
    try {
      // Send the whole batch in ONE request so the server writes once and pushes a
      // single consolidated prompt (one fetch per annotation would fire N times).
      const res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ annotations: batch }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      btn.textContent = `Sent ${batch.length} ✓`;
      console.log("[annotations] sent", batch.length, "annotation(s)");
    } catch (err) {
      pending.unshift(...batch); // put them back so nothing is lost
      btn.textContent = "Send failed ✗";
      console.error("[annotations] send failed:", err);
    } finally {
      btn.disabled = false;
      persist(); // empty after a clean send; restored batch after a failure
      setTimeout(updateButton, 1200);
    }
  };

  // Draggable: pointer-drag to reposition; a genuine click still sends. A small
  // threshold keeps pointer jitter from being treated as a drag.
  let dragging = false;
  let offX = 0, offY = 0, startX = 0, startY = 0;
  btn.addEventListener("pointerdown", (e) => {
    dragging = true;
    moved = false;
    startX = e.clientX;
    startY = e.clientY;
    const r = btn.getBoundingClientRect();
    offX = e.clientX - r.left;
    offY = e.clientY - r.top;
    btn.style.cursor = "grabbing";
    btn.setPointerCapture(e.pointerId);
  });
  btn.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    if (!moved && Math.hypot(e.clientX - startX, e.clientY - startY) < 4) return;
    moved = true;
    btn.style.left = e.clientX - offX + "px";
    btn.style.top = e.clientY - offY + "px";
    btn.style.right = "auto";
    btn.style.bottom = "auto";
  });
  btn.addEventListener("pointerup", (e) => {
    dragging = false;
    btn.style.cursor = "grab";
    if (btn.hasPointerCapture(e.pointerId)) btn.releasePointerCapture(e.pointerId);
  });

  document.body.appendChild(btn);
  updateButton();
  console.info(
    "[annotations] Alt+click any element to annotate. Inspect the queue: window.annotations"
  );
})();
