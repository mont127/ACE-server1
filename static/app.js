// IMPORTANT: we render text using textContent (not innerHTML) to avoid XSS.

const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('sendBtn');
const clearBtn = document.getElementById('clearBtn');
const modelLine = document.getElementById('modelLine');

const authBtn = document.getElementById('authBtn');
const logoutBtn = document.getElementById('logoutBtn');

const authModal = document.getElementById('authModal');
const authClose = document.getElementById('authClose');
const tabLogin = document.getElementById('tabLogin');
const tabSignup = document.getElementById('tabSignup');
const authSubmit = document.getElementById('authSubmit');
const authUser = document.getElementById('authUser');
const authPass = document.getElementById('authPass');
const authError = document.getElementById('authError');

let authMode = 'login'; // or 'signup'
let busy = false;

function el(tag, cls) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  return n;
}

function addMsg(role, text) {
  const wrap = el('div', `msg ${role}`);
  const head = el('div', 'msg-head');
  head.textContent = role === 'user' ? 'YOU' : 'ACE';

  const body = el('div', 'msg-body');
  body.textContent = text; // <-- XSS-safe

  wrap.appendChild(head);
  wrap.appendChild(body);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

function setBusy(v) {
  busy = v;
  sendBtn.disabled = v;
  sendBtn.textContent = v ? 'Thinking…' : 'Send';
}

async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const r = await fetch(path, opts);
  const ct = r.headers.get('content-type') || '';
  const data = ct.includes('application/json') ? await r.json() : await r.text();
  if (!r.ok) {
    const msg = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : 'Request failed');
    throw new Error(msg);
  }
  return data;
}

async function refreshAuthState() {
  try {
    const me = await api('/api/auth/me');
    if (me.logged_in) {
      authBtn.classList.add('hidden');
      logoutBtn.classList.remove('hidden');
      modelLine.textContent = `${modelLine.textContent} | User: ${me.username}`;
    } else {
      authBtn.classList.remove('hidden');
      logoutBtn.classList.add('hidden');
    }
  } catch (e) {
    // ignore
  }
}

function openAuth() {
  authError.textContent = '';
  authModal.classList.remove('hidden');
  authModal.setAttribute('aria-hidden', 'false');
  authUser.focus();
}

function closeAuth() {
  authModal.classList.add('hidden');
  authModal.setAttribute('aria-hidden', 'true');
}

function setAuthMode(mode) {
  authMode = mode;
  authError.textContent = '';
  if (mode === 'login') {
    tabLogin.classList.add('active');
    tabSignup.classList.remove('active');
    authSubmit.textContent = 'Login';
    authPass.autocomplete = 'current-password';
  } else {
    tabSignup.classList.add('active');
    tabLogin.classList.remove('active');
    authSubmit.textContent = 'Sign up';
    authPass.autocomplete = 'new-password';
  }
}

async function submitAuth() {
  authError.textContent = '';
  const username = authUser.value.trim();
  const password = authPass.value;
  if (!username || !password) {
    authError.textContent = 'Enter username and password.';
    return;
  }

  try {
    if (authMode === 'login') {
      await api('/api/auth/login', 'POST', { username, password });
    } else {
      await api('/api/auth/signup', 'POST', { username, password });
    }
    closeAuth();
    location.reload();
  } catch (e) {
    authError.textContent = e.message;
  }
}

async function send() {
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;

  addMsg('user', text);
  input.value = '';
  setBusy(true);

  try {
    const out = await api('/api/chat', 'POST', { prompt: text });
    addMsg('ace', out.response);
  } catch (e) {
    addMsg('ace', `[ERROR] ${e.message}`);
  } finally {
    setBusy(false);
  }
}

// Events
sendBtn.addEventListener('click', send);
clearBtn.addEventListener('click', () => { chat.innerHTML = ''; });
input.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

authBtn.addEventListener('click', openAuth);
authClose.addEventListener('click', closeAuth);
authModal.addEventListener('click', (e) => {
  if (e.target === authModal) closeAuth();
});

tabLogin.addEventListener('click', () => setAuthMode('login'));
tabSignup.addEventListener('click', () => setAuthMode('signup'));
authSubmit.addEventListener('click', submitAuth);

authPass.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    submitAuth();
  }
});

logoutBtn.addEventListener('click', async () => {
  try {
    await api('/api/auth/logout', 'POST');
    location.reload();
  } catch (e) {
    alert(e.message);
  }
});

// Boot
(async () => {
  try {
    const h = await api('/api/health');
    modelLine.textContent = `Model: ${h.model} | Device: ${h.device}`;
  } catch {
    modelLine.textContent = 'Model: (unavailable)';
  }

  setAuthMode('login');
  await refreshAuthState();

  // If not logged in, open auth modal immediately.
  try {
    const me = await api('/api/auth/me');
    if (!me.logged_in) openAuth();
  } catch {
    openAuth();
  }
})();