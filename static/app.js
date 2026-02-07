/* ACE Web UI client (vanilla JS) — GUEST ONLY
   - NO login/signup
   - Calls /api/chat directly
   - Uses a stable session_id stored in localStorage
*/

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);

  const elHealthPill = $("#healthPill");
  const elModePill = $("#modePill");
  const elQueuePill = $("#queuePill");
  const elBanner = $("#banner");

  const chatInput = $("#chatInput");
  const btnSend = $("#btnSend");
  const btnClear = $("#btnClear");
  const chatLog = $("#messages");

  function setText(el, text) {
    if (!el) return;
    el.textContent = String(text ?? "");
  }

  function showBanner(msg) {
    if (!elBanner) return;
    elBanner.textContent = String(msg ?? "");
    elBanner.classList.add("show");
  }

  function hideBanner() {
    if (!elBanner) return;
    elBanner.textContent = "";
    elBanner.classList.remove("show");
  }

  function appendMessage(role, text) {
    if (!chatLog) return;

    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;

    // Keep it simple: just text (prevents HTML injection)
    wrap.textContent = String(text ?? "");

    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function ensureSessionId() {
    const key = "ace_session_id";
    let sid = localStorage.getItem(key);
    if (!sid || sid.length < 8) {
      sid = Array.from(crypto.getRandomValues(new Uint8Array(12)))
        .map((b) => b.toString(16).padStart(2, "0"))
        .join("");
      localStorage.setItem(key, sid);
    }
    return sid;
  }

  async function apiFetch(path, opts = {}) {
    const res = await fetch(path, {
      ...opts,
      headers: {
        "Content-Type": "application/json",
        ...(opts.headers || {}),
      },
      // guest mode: no cookies required, but harmless if present
      credentials: "omit",
    });

    const ct = res.headers.get("content-type") || "";
    let data = null;

    if (ct.includes("application/json")) {
      try { data = await res.json(); } catch { data = null; }
    } else {
      try { data = await res.text(); } catch { data = null; }
    }

    if (!res.ok) {
      const detail = (data && data.detail) ? data.detail : data;
      const msg = typeof detail === "string" ? detail : JSON.stringify(detail ?? `HTTP ${res.status}`);
      const err = new Error(msg);
      err.status = res.status;
      err.data = data;
      throw err;
    }

    return data;
  }

  function updateQueuePill(health) {
    if (!elQueuePill) return;
    const maxWaiters = health?.queue_max_waiters;
    if (typeof maxWaiters === "number") {
      elQueuePill.style.display = "";
      setText(elQueuePill, `queue max: ${maxWaiters}`);
    } else {
      elQueuePill.style.display = "none";
    }
  }

  async function loadHealth() {
    try {
      const health = await apiFetch("/api/health", { method: "GET" });
      const model = health?.model || "unknown-model";
      const device = health?.device || "unknown-device";
      setText(elHealthPill, `${model} (${device})`);
      setText(elModePill, "guest");
      updateQueuePill(health);
    } catch (e) {
      setText(elHealthPill, "health error");
      showBanner(String(e.message || e));
    }
  }

  let isSending = false;

  async function sendChat(prompt) {
    if (isSending) return;
    const p = String(prompt || "").trim();
    if (!p) return;

    hideBanner();
    isSending = true;
    if (btnSend) btnSend.disabled = true;

    appendMessage("user", p);

    // typing placeholder
    const typing = document.createElement("div");
    typing.className = "msg ace";
    typing.textContent = "…";
    chatLog.appendChild(typing);
    chatLog.scrollTop = chatLog.scrollHeight;

    try {
      const session_id = ensureSessionId();
      const data = await apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({ prompt: p, session_id }),
      });

      const resp = data?.response ?? "";
      typing.remove();
      appendMessage("ace", resp);
    } catch (e) {
      typing.remove();
      appendMessage("ace", `ERROR: ${e.message || e}`);
    } finally {
      isSending = false;
      if (btnSend) btnSend.disabled = false;
    }
  }

  function clearChat() {
    if (!chatLog) return;
    chatLog.innerHTML = "";
    hideBanner();
  }

  function wire() {
    if (btnSend) {
      btnSend.addEventListener("click", async () => {
        const p = chatInput?.value ?? "";
        if (chatInput) chatInput.value = "";
        await sendChat(p);
      });
    }

    if (btnClear) {
      btnClear.addEventListener("click", () => clearChat());
    }

    if (chatInput && chatInput.tagName === "TEXTAREA") {
      chatInput.addEventListener("keydown", async (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          const p = chatInput.value;
          chatInput.value = "";
          await sendChat(p);
        }
      });
    }
  }

  document.addEventListener("DOMContentLoaded", async () => {
    wire();
    ensureSessionId();
    await loadHealth();
  });
})();