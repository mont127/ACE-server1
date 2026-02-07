/* static/app.js
   ACE Web UI — login/signup + chat
   - Uses cookie-based session from backend
   - IMPORTANT: credentials: "include" so cookies are sent on /api/chat
*/

(() => {
  // ---------- Helpers ----------
  const $ = (sel) => document.querySelector(sel);

  function esc(s) {
    return String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function getLocalSessionId() {
    let sid = localStorage.getItem("ace_session_id");
    if (!sid) {
      sid = (crypto?.randomUUID?.() || Math.random().toString(16).slice(2) + Date.now().toString(16));
      localStorage.setItem("ace_session_id", sid);
    }
    return sid;
  }

  async function api(path, { method = "GET", body = null } = {}) {
    const opts = {
      method,
      headers: { "Content-Type": "application/json" },
      credentials: "include", // <-- critical for cookie-based auth
    };
    if (body !== null) opts.body = JSON.stringify(body);

    const res = await fetch(path, opts);

    let data = null;
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) {
      data = await res.json().catch(() => null);
    } else {
      data = await res.text().catch(() => null);
    }

    if (!res.ok) {
      const msg =
        (data && typeof data === "object" && (data.detail || data.error || data.message)) ||
        (typeof data === "string" && data) ||
        `HTTP ${res.status}`;
      const err = new Error(msg);
      err.status = res.status;
      err.data = data;
      throw err;
    }

    return data;
  }

  // ---------- DOM ----------
  // Expected IDs/classes (match your index.html):
  // #authView, #chatView
  // #loginForm, #signupForm
  // #loginUser, #loginPass, #signupUser, #signupPass
  // #authError, #chatError
  // #logoutBtn
  // #healthLine
  // #chatLog
  // #promptInput, #sendBtn
  // #acwToggle (optional checkbox), #maxTokens (optional), #temp (optional), #topP (optional)
  const authView = $("#authView");
  const chatView = $("#chatView");

  const loginForm = $("#loginForm");
  const signupForm = $("#signupForm");
  const loginUser = $("#loginUser");
  const loginPass = $("#loginPass");
  const signupUser = $("#signupUser");
  const signupPass = $("#signupPass");
  const authError = $("#authError");

  const logoutBtn = $("#logoutBtn");
  const healthLine = $("#healthLine");
  const chatError = $("#chatError");

  const chatLog = $("#chatLog");
  const promptInput = $("#promptInput");
  const sendBtn = $("#sendBtn");

  const acwToggle = $("#acwToggle"); // optional
  const maxTokensEl = $("#maxTokens"); // optional
  const tempEl = $("#temp"); // optional
  const topPEl = $("#topP"); // optional

  // ---------- UI Render ----------
  function showAuth(msg = "") {
    if (authView) authView.style.display = "";
    if (chatView) chatView.style.display = "none";
    if (authError) {
      authError.textContent = msg || "";
      authError.style.display = msg ? "" : "none";
    }
    if (chatError) {
      chatError.textContent = "";
      chatError.style.display = "none";
    }
  }

  function showChat() {
    if (authView) authView.style.display = "none";
    if (chatView) chatView.style.display = "";
    if (authError) {
      authError.textContent = "";
      authError.style.display = "none";
    }
    if (chatError) {
      chatError.textContent = "";
      chatError.style.display = "none";
    }
    promptInput?.focus?.();
  }

  function setBusy(busy) {
    if (sendBtn) sendBtn.disabled = !!busy;
    if (promptInput) promptInput.disabled = !!busy;
    if (busy) {
      sendBtn && (sendBtn.textContent = "Thinking…");
    } else {
      sendBtn && (sendBtn.textContent = "Send");
    }
  }

  function appendMsg(role, text) {
    if (!chatLog) return;

    const wrap = document.createElement("div");
    wrap.className = `msg ${role}`;

    const header = document.createElement("div");
    header.className = "msgHeader";
    header.textContent = role === "user" ? "YOU" : "ACE";

    const body = document.createElement("div");
    body.className = "msgBody";
    // Safe render:
    body.innerHTML = esc(text).replace(/\n/g, "<br>");

    wrap.appendChild(header);
    wrap.appendChild(body);

    chatLog.appendChild(wrap);
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function setChatError(msg = "") {
    if (!chatError) return;
    chatError.textContent = msg || "";
    chatError.style.display = msg ? "" : "none";
  }

  // ---------- Auth ----------
  async function refreshMe() {
    // backend should return {logged_in: true, username: "..."} or 401
    try {
      const me = await api("/api/auth/me");
      if (me && (me.logged_in || me.username)) {
        showChat();
        return true;
      }
      showAuth("");
      return false;
    } catch (e) {
      showAuth("");
      return false;
    }
  }

  async function doSignup(username, password) {
    await api("/api/auth/signup", { method: "POST", body: { username, password } });
  }

  async function doLogin(username, password) {
    await api("/api/auth/login", { method: "POST", body: { username, password } });
  }

  async function doLogout() {
    try {
      await api("/api/auth/logout", { method: "POST", body: {} });
    } catch (_) {
      // ignore
    }
    showAuth("");
  }

  // ---------- Chat ----------
  function buildPrompt(raw) {
    const text = (raw || "").trim();
    if (!text) return "";

    const acwOn = !!acwToggle?.checked;
    if (acwOn && !/\s+acw$/i.test(text)) return `${text} ACW`;
    return text;
  }

  function readTuning() {
    // optional; backend may ignore these
    const max_new_tokens = maxTokensEl ? parseInt(maxTokensEl.value || "", 10) : undefined;
    const temperature = tempEl ? parseFloat(tempEl.value || "") : undefined;
    const top_p = topPEl ? parseFloat(topPEl.value || "") : undefined;

    const out = {};
    if (Number.isFinite(max_new_tokens) && max_new_tokens > 0) out.max_new_tokens = max_new_tokens;
    if (Number.isFinite(temperature) && temperature > 0) out.temperature = temperature;
    if (Number.isFinite(top_p) && top_p > 0 && top_p <= 1) out.top_p = top_p;
    return out;
  }

  async function sendChat() {
    setChatError("");

    const raw = promptInput?.value ?? "";
    const prompt = buildPrompt(raw);
    if (!prompt) return;

    promptInput.value = "";
    appendMsg("user", prompt);

    setBusy(true);

    const session_id = getLocalSessionId();
    const tuning = readTuning();

    try {
      const payload = { prompt, session_id, ...tuning };
      const out = await api("/api/chat", { method: "POST", body: payload });

      // expected: {session_id, response, device, model}
      if (out?.session_id) localStorage.setItem("ace_session_id", out.session_id);
      appendMsg("ace", out?.response ?? "[No response]");
    } catch (e) {
      const msg = e?.message || "Request failed";
      setChatError(msg);

      // If auth expired, bounce to login
      if (e?.status === 401) {
        showAuth("Session expired. Please log in again.");
      }
    } finally {
      setBusy(false);
    }
  }

  // ---------- Startup ----------
  async function loadHealth() {
    if (!healthLine) return;
    try {
      const h = await api("/api/health");
      // expected: {ok, model, device, queue_max_waiters?}
      const bits = [];
      if (h?.model) bits.push(`Model: ${h.model}`);
      if (h?.device) bits.push(`Device: ${h.device}`);
      if (h?.queue_max_waiters !== undefined) bits.push(`Queue: ${h.queue_max_waiters}`);
      healthLine.textContent = bits.join(" • ") || "OK";
    } catch (_) {
      healthLine.textContent = "Health check failed";
    }
  }

  function wireEvents() {
    loginForm?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!loginUser?.value || !loginPass?.value) return showAuth("Enter username + password.");
      try {
        showAuth("");
        await doLogin(loginUser.value.trim(), loginPass.value);
        await refreshMe();
      } catch (e) {
        showAuth(e?.message || "Login failed");
      }
    });

    signupForm?.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      if (!signupUser?.value || !signupPass?.value) return showAuth("Enter username + password.");
      try {
        showAuth("");
        await doSignup(signupUser.value.trim(), signupPass.value);
        // auto-login after signup
        await doLogin(signupUser.value.trim(), signupPass.value);
        await refreshMe();
      } catch (e) {
        showAuth(e?.message || "Signup failed");
      }
    });

    logoutBtn?.addEventListener("click", async () => {
      await doLogout();
    });

    sendBtn?.addEventListener("click", sendChat);

    promptInput?.addEventListener("keydown", (ev) => {
      // Enter to send, Shift+Enter for newline
      if (ev.key === "Enter" && !ev.shiftKey) {
        ev.preventDefault();
        sendChat();
      }
    });
  }

  async function init() {
    wireEvents();
    await loadHealth();
    await refreshMe();

    // tiny delay so UI feels stable
    await sleep(50);
    promptInput?.focus?.();
  }

  init();
})();