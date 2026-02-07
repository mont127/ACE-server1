/*
  ACE UI - resilient client
  - Works even if element IDs/classes differ slightly
  - Uses event delegation for all buttons/links
  - Stores auth via cookie (credentials: 'include')
  - Renders text safely (textContent) to avoid XSS
*/

const API_BASE = ""; // same-origin

// ----------------------------
// Helpers
// ----------------------------

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function byIdLoose(...ids) {
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) return el;
  }
  return null;
}

function firstExisting(selectors) {
  for (const s of selectors) {
    const el = $(s);
    if (el) return el;
  }
  return null;
}

function setText(el, txt) {
  if (!el) return;
  el.textContent = txt;
}

function show(el) {
  if (!el) return;
  el.classList.remove("hidden");
  el.style.display = "";
  el.setAttribute("aria-hidden", "false");
}

function hide(el) {
  if (!el) return;
  el.classList.add("hidden");
  el.style.display = "none";
  el.setAttribute("aria-hidden", "true");
}

function isVisible(el) {
  if (!el) return false;
  return !(el.classList.contains("hidden") || getComputedStyle(el).display === "none");
}

function toast(msg) {
  // Try to use existing toast/status area if present
  const status = firstExisting([
    "#status",
    "#toast",
    ".status",
    ".toast",
    "[data-role='status']",
  ]);
  if (status) {
    setText(status, msg);
    show(status);
    setTimeout(() => {
      // don't hide permanently if page uses it
      if (status.id === "toast" || status.classList.contains("toast")) hide(status);
    }, 3500);
    return;
  }
  alert(msg);
}

async function api(path, opts = {}) {
  const res = await fetch(API_BASE + path, {
    ...opts,
    headers: {
      "Content-Type": "application/json",
      ...(opts.headers || {}),
    },
    credentials: "include",
  });

  let data = null;
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    data = await res.json().catch(() => null);
  } else {
    data = await res.text().catch(() => null);
  }

  if (!res.ok) {
    const detail = (data && data.detail) ? data.detail : (typeof data === "string" ? data : "Request failed");
    const err = new Error(detail);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

// ----------------------------
// DOM discovery (tolerant)
// ----------------------------

const ui = {
  // Chat
  chatForm: firstExisting(["#chatForm", "form#chat", "form[data-role='chat']"]),
  chatInput: firstExisting([
    "#prompt",
    "#chatInput",
    "textarea#prompt",
    "textarea[name='prompt']",
    "textarea[data-role='prompt']",
    "input#prompt",
    "input[name='prompt']",
  ]),
  sendBtn: firstExisting([
    "#sendBtn",
    "button#send",
    "button[type='submit'][data-role='send']",
    "button[data-action='send']",
  ]),
  messages: firstExisting([
    "#messages",
    "#chatMessages",
    ".messages",
    "[data-role='messages']",
  ]),

  // Auth modal + pieces
  authModal: firstExisting([
    "#authModal",
    "#modal",
    ".modal",
    "[data-role='auth-modal']",
  ]),
  authBackdrop: firstExisting([
    "#authBackdrop",
    ".modal-backdrop",
    "#backdrop",
    "[data-role='auth-backdrop']",
  ]),
  authClose: firstExisting([
    "#authClose",
    "#closeModal",
    "button[aria-label='Close']",
    "button[data-action='close-auth']",
    ".modal .close",
  ]),

  loginForm: firstExisting([
    "#loginForm",
    "form#login",
    "form[data-role='login']",
  ]),
  signupForm: firstExisting([
    "#signupForm",
    "form#signup",
    "form[data-role='signup']",
  ]),

  // top buttons
  openLoginBtn: firstExisting([
    "#openLogin",
    "#loginBtn",
    "button[data-action='open-login']",
    "a[data-action='open-login']",
  ]),
  openSignupBtn: firstExisting([
    "#openSignup",
    "#signupBtn",
    "button[data-action='open-signup']",
    "a[data-action='open-signup']",
  ]),
  logoutBtn: firstExisting([
    "#logoutBtn",
    "button[data-action='logout']",
    "a[data-action='logout']",
  ]),

  // Areas to toggle
  authedOnly: $$("[data-auth='in']"),
  unauthedOnly: $$("[data-auth='out']"),

  // Debug box (optional)
  debug: firstExisting(["#debug", "pre#debug", "[data-role='debug']"]),
};

function debugLog(...args) {
  console.log("[ACE UI]", ...args);
  if (ui.debug) {
    ui.debug.textContent += args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(" ") + "\n";
  }
}

// ----------------------------
// Modal control
// ----------------------------

function openAuth(mode) {
  // mode: 'login' | 'signup'
  if (!ui.authModal) {
    toast("Auth UI missing: auth modal not found in DOM");
    debugLog("No authModal found. Selectors may not match your index.html.");
    return;
  }

  show(ui.authModal);
  if (ui.authBackdrop) show(ui.authBackdrop);

  // Toggle forms if both exist
  if (ui.loginForm && ui.signupForm) {
    if (mode === "signup") {
      show(ui.signupForm);
      hide(ui.loginForm);
    } else {
      show(ui.loginForm);
      hide(ui.signupForm);
    }
  }

  // Focus first input
  const focusTarget = firstExisting([
    "#loginEmail",
    "#signupEmail",
    "input[type='email']",
    "input[name='email']",
  ], ui.authModal);
  if (focusTarget) focusTarget.focus();
}

function closeAuth() {
  hide(ui.authModal);
  if (ui.authBackdrop) hide(ui.authBackdrop);
}

// ----------------------------
// Chat rendering
// ----------------------------

function appendMsg(role, text) {
  if (!ui.messages) return;
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;

  const head = document.createElement("div");
  head.className = "msg-head";
  head.textContent = role === "user" ? "YOU" : "ACE";

  const body = document.createElement("div");
  body.className = "msg-body";
  body.textContent = text; // SAFE

  wrap.appendChild(head);
  wrap.appendChild(body);
  ui.messages.appendChild(wrap);
  ui.messages.scrollTop = ui.messages.scrollHeight;
}

function setChatEnabled(enabled) {
  if (ui.chatInput) ui.chatInput.disabled = !enabled;
  if (ui.sendBtn) ui.sendBtn.disabled = !enabled;
}

// ----------------------------
// Auth state
// ----------------------------

let authState = { loggedIn: false, user: null };

function applyAuthUI() {
  // If markup doesn't use these, harmless.
  ui.authedOnly.forEach(el => (authState.loggedIn ? show(el) : hide(el)));
  ui.unauthedOnly.forEach(el => (authState.loggedIn ? hide(el) : show(el)));
  setChatEnabled(authState.loggedIn);
}

async function refreshMe() {
  try {
    const me = await api("/api/auth/me", { method: "GET" });
    authState.loggedIn = !!me.logged_in;
    authState.user = me.user || null;
    debugLog("/api/auth/me", me);
  } catch (e) {
    debugLog("/api/auth/me failed", e);
    authState.loggedIn = false;
    authState.user = null;
  }
  applyAuthUI();
}

// ----------------------------
// Event wiring (delegation)
// ----------------------------

function matchesAny(el, selectors) {
  return selectors.some(s => {
    try { return el.matches(s); } catch { return false; }
  });
}

function findClickableRole(el) {
  // climb up to allow clicks on inner spans/icons
  let cur = el;
  for (let i = 0; i < 4 && cur; i++) {
    if (cur.nodeType === 1) {
      const id = (cur.id || "").toLowerCase();
      const act = (cur.getAttribute("data-action") || "").toLowerCase();
      const role = (cur.getAttribute("data-role") || "").toLowerCase();
      const txt = (cur.textContent || "").trim().toLowerCase();

      // explicit actions
      if (act === "open-login") return { type: "open-login", el: cur };
      if (act === "open-signup") return { type: "open-signup", el: cur };
      if (act === "logout") return { type: "logout", el: cur };
      if (act === "close-auth") return { type: "close-auth", el: cur };

      // IDs
      if (["openlogin", "loginbtn", "login"].includes(id)) return { type: "open-login", el: cur };
      if (["opensignup", "signupbtn", "signup", "register"].includes(id)) return { type: "open-signup", el: cur };
      if (["logoutbtn", "logout"].includes(id)) return { type: "logout", el: cur };
      if (["authclose", "closemodal", "close"].includes(id)) return { type: "close-auth", el: cur };

      // roles
      if (role === "open-login") return { type: "open-login", el: cur };
      if (role === "open-signup") return { type: "open-signup", el: cur };

      // text fallback
      if (cur.tagName === "BUTTON" || cur.tagName === "A") {
        if (txt === "login" || txt === "log in") return { type: "open-login", el: cur };
        if (txt === "signup" || txt === "sign up" || txt === "register") return { type: "open-signup", el: cur };
        if (txt === "logout" || txt === "log out") return { type: "logout", el: cur };
      }
    }
    cur = cur.parentElement;
  }
  return null;
}

document.addEventListener("click", async (ev) => {
  const hit = findClickableRole(ev.target);
  if (!hit) return;

  ev.preventDefault();

  if (hit.type === "open-login") {
    debugLog("click open-login");
    openAuth("login");
    return;
  }
  if (hit.type === "open-signup") {
    debugLog("click open-signup");
    openAuth("signup");
    return;
  }
  if (hit.type === "close-auth") {
    debugLog("click close-auth");
    closeAuth();
    return;
  }
  if (hit.type === "logout") {
    debugLog("click logout");
    try {
      await api("/api/auth/logout", { method: "POST", body: "{}" });
    } catch (e) {
      debugLog("logout error", e);
    }
    await refreshMe();
    toast("Logged out");
    return;
  }
});

// Backdrop click closes
if (ui.authBackdrop) {
  ui.authBackdrop.addEventListener("click", () => {
    if (ui.authModal && isVisible(ui.authModal)) closeAuth();
  });
}

// ESC closes
document.addEventListener("keydown", (ev) => {
  if (ev.key === "Escape" && ui.authModal && isVisible(ui.authModal)) {
    closeAuth();
  }
});

// Forms: tolerate different input IDs
function getVal(selectors, root) {
  const el = firstExisting(selectors, root);
  return el ? String(el.value || "").trim() : "";
}

async function doLogin(formEl) {
  const email = getVal(["#loginEmail", "input[name='email']", "input[type='email']"], formEl);
  const password = getVal(["#loginPassword", "input[name='password']", "input[type='password']"], formEl);

  if (!email || !password) {
    toast("Email and password required");
    return;
  }

  try {
    await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    closeAuth();
    await refreshMe();
    toast("Logged in");
  } catch (e) {
    debugLog("login error", e);
    toast(`Login failed: ${e.message}`);
  }
}

async function doSignup(formEl) {
  const email = getVal(["#signupEmail", "input[name='email']", "input[type='email']"], formEl);
  const password = getVal(["#signupPassword", "input[name='password']", "input[type='password']"], formEl);

  if (!email || !password) {
    toast("Email and password required");
    return;
  }

  try {
    await api("/api/auth/signup", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    // auto-login after signup
    await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    closeAuth();
    await refreshMe();
    toast("Account created");
  } catch (e) {
    debugLog("signup error", e);
    toast(`Signup failed: ${e.message}`);
  }
}

// Attach submit listeners if forms exist
if (ui.loginForm) {
  ui.loginForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    doLogin(ui.loginForm);
  });
}

if (ui.signupForm) {
  ui.signupForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    doSignup(ui.signupForm);
  });
}

// If buttons exist inside modal (some UIs use button click instead of submit)
if (ui.authModal) {
  ui.authModal.addEventListener("click", (ev) => {
    const t = ev.target;
    if (!(t instanceof Element)) return;

    // Any button with data-action
    const actEl = t.closest("[data-action]");
    const act = actEl ? (actEl.getAttribute("data-action") || "").toLowerCase() : "";
    if (act === "do-login") {
      ev.preventDefault();
      doLogin(ui.loginForm || ui.authModal);
    }
    if (act === "do-signup") {
      ev.preventDefault();
      doSignup(ui.signupForm || ui.authModal);
    }

    // Text fallback
    const btn = t.closest("button");
    const txt = (btn ? btn.textContent : "").trim().toLowerCase();
    if (txt === "login" || txt === "log in") {
      // If button is inside a form, submit handler already runs.
      // If not, run explicitly.
      if (!btn || !btn.closest("form")) {
        ev.preventDefault();
        doLogin(ui.loginForm || ui.authModal);
      }
    }
    if (txt === "signup" || txt === "sign up" || txt === "register") {
      if (!btn || !btn.closest("form")) {
        ev.preventDefault();
        doSignup(ui.signupForm || ui.authModal);
      }
    }
  });
}

// ----------------------------
// Chat submit
// ----------------------------

async function sendChat() {
  const prompt = (ui.chatInput ? ui.chatInput.value : "").trim();
  if (!prompt) return;

  if (!authState.loggedIn) {
    openAuth("login");
    toast("Please log in first");
    return;
  }

  appendMsg("user", prompt);
  if (ui.chatInput) ui.chatInput.value = "";

  setChatEnabled(false);
  try {
    const out = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ prompt }),
    });
    appendMsg("ace", out.response || "");
  } catch (e) {
    debugLog("chat error", e);
    toast(`Chat failed: ${e.message}`);
  } finally {
    setChatEnabled(true);
  }
}

// If your UI has a form, hook submit
if (ui.chatForm) {
  ui.chatForm.addEventListener("submit", (ev) => {
    ev.preventDefault();
    sendChat();
  });
}

// If it doesn't, hook send button
if (ui.sendBtn) {
  ui.sendBtn.addEventListener("click", (ev) => {
    ev.preventDefault();
    sendChat();
  });
}

// Enter-to-send for textarea/input if desired (only if not multi-line textarea)
if (ui.chatInput) {
  ui.chatInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) {
      // If it's a textarea we still treat Enter as send unless Shift
      ev.preventDefault();
      sendChat();
    }
  });
}

// ----------------------------
// Boot
// ----------------------------

window.addEventListener("load", async () => {
  debugLog("boot", {
    authModal: !!ui.authModal,
    loginForm: !!ui.loginForm,
    signupForm: !!ui.signupForm,
    chatInput: !!ui.chatInput,
  });

  // If UI lacks explicit buttons, it still works via delegation.
  // Make chat disabled until logged in.
  setChatEnabled(false);
  await refreshMe();
});
