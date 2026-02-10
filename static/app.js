/* ACE Web UI client (vanilla JS)
   - Login/Signup (cookie session)
   - Loads chat history from /api/chat/history
   - Calls /api/chat with credentials
*/

(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);

  const elHealthPill = $("#healthPill");
  const elAuthPill = $("#authPill");
  const elQueuePill = $("#queuePill");
  const elBanner = $("#banner");

  const chatInput = $("#chatInput");
  const btnSend = $("#btnSend");
  const btnClear = $("#btnClear");
  const chatLog = $("#messages");

  const btnLogin = $("#btnLogin");
  const btnSignup = $("#btnSignup");
  const btnLogout = $("#btnLogout");

  const loginBackdrop = $("#loginBackdrop");
  const loginClose = $("#loginClose");
  const loginForm = $("#loginForm");
  const loginIdent = $("#loginIdent");
  const loginPassword = $("#loginPassword");
  const loginError = $("#loginError");

  const signupBackdrop = $("#signupBackdrop");
  const signupClose = $("#signupClose");
  const signupForm = $("#signupForm");
  const signupEmail = $("#signupEmail");
  const signupPassword = $("#signupPassword");
  const signupError = $("#signupError");

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

  function showModal(backdrop) {
    if (!backdrop) return;
    backdrop.classList.add("show");
    backdrop.setAttribute("aria-hidden", "false");
  }

  function hideModal(backdrop) {
    if (!backdrop) return;
    backdrop.classList.remove("show");
    backdrop.setAttribute("aria-hidden", "true");
  }

  function appendMessage(role, text) {
    if (!chatLog) return;
    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;
    wrap.textContent = String(text ?? "");
    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function clearChatUI() {
    if (!chatLog) return;
    chatLog.innerHTML = "";
    hideBanner();
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
      credentials: "include",
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
      updateQueuePill(health);
    } catch (e) {
      setText(elHealthPill, "health error");
      showBanner(String(e.message || e));
    }
  }

  let me = { logged_in: false, username: "" };

  function renderAuth() {
    if (me?.logged_in) {
      setText(elAuthPill, `logged in: ${me.username || "user"}`);
      if (btnLogin) btnLogin.style.display = "none";
      if (btnSignup) btnSignup.style.display = "none";
      if (btnLogout) btnLogout.style.display = "";
    } else {
      setText(elAuthPill, "not logged in");
      if (btnLogin) btnLogin.style.display = "";
      if (btnSignup) btnSignup.style.display = "";
      if (btnLogout) btnLogout.style.display = "none";
    }
  }

  async function refreshMe() {
    try {
      const data = await apiFetch("/api/auth/me", { method: "GET" });
      me = data || { logged_in: false };
    } catch {
      me = { logged_in: false };
    }
    renderAuth();
  }

  async function loadHistory() {
    if (!me?.logged_in) return;
    try {
      const data = await apiFetch("/api/chat/history?limit=200", { method: "GET" });
      clearChatUI();
      const msgs = data?.messages || [];
      for (const m of msgs) {
        const role = (m.role === "assistant") ? "ace" : "user";
        appendMessage(role, m.content);
      }
    } catch (e) {
      showBanner(String(e.message || e));
    }
  }

  function buildAuthPayload(identOrEmail, password) {
    const v = String(identOrEmail || "").trim();
    // backend accepts username OR email
    return { username: v, email: v, password: String(password || "") };
  }

  async function doLogin() {
    setText(loginError, "");
    try {
      await apiFetch("/api/auth/login", {
        method: "POST",
        body: JSON.stringify(buildAuthPayload(loginIdent?.value, loginPassword?.value)),
      });
      hideModal(loginBackdrop);
      await refreshMe();
      await loadHistory();
    } catch (e) {
      setText(loginError, `Login failed: ${e.message || e}`);
      throw e;
    }
  }

  async function doSignup() {
    setText(signupError, "");
    try {
      await apiFetch("/api/auth/signup", {
        method: "POST",
        body: JSON.stringify({
          email: String(signupEmail?.value || "").trim(),
          password: String(signupPassword?.value || ""),
        }),
      });
      hideModal(signupBackdrop);
      await refreshMe();
      await loadHistory();
    } catch (e) {
      setText(signupError, `Signup failed: ${e.message || e}`);
      throw e;
    }
  }

  async function doLogout() {
    try {
      await apiFetch("/api/auth/logout", { method: "POST" });
    } catch {
      // ignore
    }
    me = { logged_in: false };
    renderAuth();
    clearChatUI();
  }

  let isSending = false;

  async function sendChat(prompt) {
    if (isSending) return;
    const p = String(prompt || "").trim();
    if (!p) return;

    if (!me?.logged_in) {
      showBanner("Please login to use chat.");
      showModal(loginBackdrop);
      return;
    }

    hideBanner();
    isSending = true;
    if (btnSend) btnSend.disabled = true;

    appendMessage("user", p);

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
      if (e.status === 401) {
        showBanner("Session expired. Please login again.");
        showModal(loginBackdrop);
      } else {
        appendMessage("ace", `ERROR: ${e.message || e}`);
      }
    } finally {
      isSending = false;
      if (btnSend) btnSend.disabled = false;
    }
  }

  async function clearServerHistory() {
    if (!me?.logged_in) {
      clearChatUI();
      return;
    }
    try {
      await apiFetch("/api/chat/clear", { method: "POST" });
    } catch (e) {
      showBanner(String(e.message || e));
    }
    clearChatUI();
  }

  function wire() {
    if (btnLogin) btnLogin.addEventListener("click", () => showModal(loginBackdrop));
    if (btnSignup) btnSignup.addEventListener("click", () => showModal(signupBackdrop));
    if (btnLogout) btnLogout.addEventListener("click", () => doLogout());

    if (loginClose) loginClose.addEventListener("click", () => hideModal(loginBackdrop));
    if (signupClose) signupClose.addEventListener("click", () => hideModal(signupBackdrop));

    if (loginBackdrop) loginBackdrop.addEventListener("click", (e) => {
      if (e.target === loginBackdrop) hideModal(loginBackdrop);
    });
    if (signupBackdrop) signupBackdrop.addEventListener("click", (e) => {
      if (e.target === signupBackdrop) hideModal(signupBackdrop);
    });

    if (loginForm) {
      loginForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await doLogin();
      });
    }

    if (signupForm) {
      signupForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        await doSignup();
      });
    }

    if (btnSend) {
      btnSend.addEventListener("click", async () => {
        const p = chatInput?.value ?? "";
        if (chatInput) chatInput.value = "";
        await sendChat(p);
      });
    }

    if (btnClear) {
      btnClear.addEventListener("click", async () => {
        await clearServerHistory();
      });
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
    hideModal(loginBackdrop);
    hideModal(signupBackdrop);
    await loadHealth();
    await refreshMe();
    if (me?.logged_in) {
      await loadHistory();
    }
  });
})();