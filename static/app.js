/*
  ACE Web UI - app.js (fixed)
  - Defensive DOM binding (won't crash if an element is missing)
  - Login / Signup modal works
  - Chat works only when logged in
  - Uses cookies via fetch(..., { credentials: 'include' })
  - Minimal XSS safety: render model output as textContent (never innerHTML)
*/

(() => {
  'use strict';

  // -------------------------
  // Helpers
  // -------------------------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  function safeText(el, text) {
    if (!el) return;
    el.textContent = String(text ?? '');
  }

  function show(el) {
    if (!el) return;
    el.classList.remove('hidden');
    el.style.display = '';
  }

  function hide(el) {
    if (!el) return;
    el.classList.add('hidden');
    el.style.display = 'none';
  }

  function setDisabled(el, disabled) {
    if (!el) return;
    el.disabled = !!disabled;
    if (disabled) el.setAttribute('aria-disabled', 'true');
    else el.removeAttribute('aria-disabled');
  }

  function toast(msg, kind = 'info') {
    // If your HTML has a toast element, use it; otherwise fallback to console + alert for errors.
    const t = $('#toast') || $('#status') || $('#banner');
    if (t) {
      t.classList.remove('hidden');
      t.dataset.kind = kind;
      safeText(t, msg);
      // auto-hide after a bit
      clearTimeout(toast._tm);
      toast._tm = setTimeout(() => {
        // do not hide persistent banners if user wants to keep them; ok to hide here.
        t.classList.add('hidden');
      }, 3500);
    } else {
      if (kind === 'error') {
        console.error(msg);
        // avoid spamming alerts; only alert for error.
        alert(msg);
      } else {
        console.log(msg);
      }
    }
  }

  async function api(path, { method = 'GET', body = null, headers = {} } = {}) {
    const opts = {
      method,
      credentials: 'include',
      headers: {
        ...headers,
      },
    };

    if (body !== null) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }

    const res = await fetch(path, opts);
    const ct = res.headers.get('content-type') || '';
    const isJson = ct.includes('application/json');
    const data = isJson ? await res.json().catch(() => ({})) : await res.text().catch(() => '');

    if (!res.ok) {
      const detail = (data && data.detail) ? data.detail : (typeof data === 'string' ? data : 'Request failed');
      const err = new Error(detail);
      err.status = res.status;
      err.data = data;
      throw err;
    }

    return data;
  }

  // -------------------------
  // State
  // -------------------------
  const state = {
    user: null,
    sessionId: null,
    busy: false,
  };

  // -------------------------
  // DOM refs (optional)
  // -------------------------
  // Modal
  const modal = $('#authModal') || $('#modal') || $('.modal');
  const modalBackdrop = $('#authBackdrop') || $('#modalBackdrop') || $('.modal-backdrop');
  const modalCloseBtn = $('#authClose') || $('#modalClose') || $('#closeModal') || $('.modal-close');

  // Tabs / forms
  const tabLoginBtn = $('#tabLogin') || $('[data-tab="login"]');
  const tabSignupBtn = $('#tabSignup') || $('[data-tab="signup"]');

  const loginForm = $('#loginForm') || $('#formLogin') || $('[data-form="login"]');
  const signupForm = $('#signupForm') || $('#formSignup') || $('[data-form="signup"]');

  // Buttons
  const openAuthBtn = $('#openAuth') || $('#loginBtn') || $('#btnLogin') || $('[data-action="open-auth"]');
  const logoutBtn = $('#logoutBtn') || $('#btnLogout') || $('[data-action="logout"]');

  // Status area
  const whoamiEl = $('#whoami') || $('#userLabel') || $('#userEmail');
  const authHintEl = $('#authHint') || $('#authStatus');

  // Chat
  const chatForm = $('#chatForm') || $('#formChat') || $('[data-form="chat"]');
  const chatInput = $('#chatInput') || $('#prompt') || $('#input') || $('textarea');
  const sendBtn = $('#sendBtn') || $('#btnSend') || $('[data-action="send"]');
  const chatLog = $('#chatLog') || $('#messages') || $('#chat') || $('#output');

  // -------------------------
  // Modal control
  // -------------------------
  function openModal() {
    if (!modal) {
      toast('Auth UI not found in index.html (missing #authModal).', 'error');
      return;
    }
    show(modal);
    if (modalBackdrop) show(modalBackdrop);
    // default to login tab
    selectTab('login');
  }

  function closeModal() {
    if (!modal) return;
    hide(modal);
    if (modalBackdrop) hide(modalBackdrop);
  }

  function selectTab(which) {
    const w = (which || 'login').toLowerCase();

    if (loginForm) {
      if (w === 'login') show(loginForm); else hide(loginForm);
    }
    if (signupForm) {
      if (w === 'signup') show(signupForm); else hide(signupForm);
    }

    // Optional tab button styles
    if (tabLoginBtn) tabLoginBtn.classList.toggle('active', w === 'login');
    if (tabSignupBtn) tabSignupBtn.classList.toggle('active', w === 'signup');
  }

  // -------------------------
  // UI updates
  // -------------------------
  function setLoggedIn(user) {
    state.user = user || null;
    const loggedIn = !!state.user;

    if (loggedIn) {
      safeText(whoamiEl, state.user.email || state.user.username || 'Logged in');
      safeText(authHintEl, '');
      if (openAuthBtn) hide(openAuthBtn);
      if (logoutBtn) show(logoutBtn);
      if (chatInput) chatInput.placeholder = 'Ask ACE…';
      setDisabled(chatInput, false);
      setDisabled(sendBtn, false);
      closeModal();
    } else {
      safeText(whoamiEl, 'Guest');
      safeText(authHintEl, 'Log in to chat');
      if (openAuthBtn) show(openAuthBtn);
      if (logoutBtn) hide(logoutBtn);
      if (chatInput) chatInput.placeholder = 'Log in to chat…';
      setDisabled(chatInput, true);
      setDisabled(sendBtn, true);
    }
  }

  function appendMessage(role, text) {
    if (!chatLog) return;

    const line = document.createElement('div');
    line.className = `msg ${role}`;

    const header = document.createElement('div');
    header.className = 'msg-head';
    header.textContent = role === 'user' ? 'YOU' : 'ACE';

    const body = document.createElement('pre');
    body.className = 'msg-body';
    // IMPORTANT: avoid XSS by using textContent
    body.textContent = String(text ?? '');

    line.appendChild(header);
    line.appendChild(body);
    chatLog.appendChild(line);

    // scroll to bottom
    chatLog.scrollTop = chatLog.scrollHeight;
  }

  function setBusy(b) {
    state.busy = !!b;
    setDisabled(sendBtn, b || !state.user);
    if (chatInput) setDisabled(chatInput, b || !state.user);
    if (sendBtn) sendBtn.textContent = b ? 'Thinking…' : 'Send';
  }

  // -------------------------
  // Auth actions
  // -------------------------
  async function refreshMe() {
    try {
      const me = await api('/api/auth/me');
      // Expected shapes: { logged_in: true, user: {..} } OR { user: {..} } OR { logged_in:false }
      if (me && (me.logged_in === false || me.user === null)) {
        setLoggedIn(null);
        return;
      }
      if (me && me.user) {
        setLoggedIn(me.user);
        return;
      }
      // fallback: if endpoint returns user directly
      if (me && (me.email || me.username)) {
        setLoggedIn(me);
        return;
      }
      setLoggedIn(null);
    } catch (e) {
      // If /api/auth/me not implemented or errors, treat as logged out
      console.warn('refreshMe failed:', e);
      setLoggedIn(null);
    }
  }

  function getFormVal(form, name, fallbackSel) {
    if (!form) return '';
    const byName = form.querySelector(`[name="${name}"]`);
    if (byName) return (byName.value || '').trim();
    const bySel = fallbackSel ? form.querySelector(fallbackSel) : null;
    if (bySel) return (bySel.value || '').trim();
    return '';
  }

  async function doSignup(ev) {
    ev?.preventDefault?.();
    if (!signupForm) {
      toast('Signup form not found in HTML.', 'error');
      return;
    }

    const email = getFormVal(signupForm, 'email', '#signupEmail');
    const password = getFormVal(signupForm, 'password', '#signupPassword');

    if (!email || !password) {
      toast('Enter email + password.', 'error');
      return;
    }

    try {
      setBusy(true);
      await api('/api/auth/signup', { method: 'POST', body: { email, password } });
      toast('Account created. Logging in…');
      await api('/api/auth/login', { method: 'POST', body: { email, password } });
      await refreshMe();
    } catch (e) {
      toast(`Signup failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }

  async function doLogin(ev) {
    ev?.preventDefault?.();
    if (!loginForm) {
      toast('Login form not found in HTML.', 'error');
      return;
    }

    const email = getFormVal(loginForm, 'email', '#loginEmail');
    const password = getFormVal(loginForm, 'password', '#loginPassword');

    if (!email || !password) {
      toast('Enter email + password.', 'error');
      return;
    }

    try {
      setBusy(true);
      await api('/api/auth/login', { method: 'POST', body: { email, password } });
      await refreshMe();
    } catch (e) {
      toast(`Login failed: ${e.message}`, 'error');
    } finally {
      setBusy(false);
    }
  }

  async function doLogout(ev) {
    ev?.preventDefault?.();
    try {
      setBusy(true);
      await api('/api/auth/logout', { method: 'POST' });
    } catch (e) {
      // If not implemented, still clear UI
      console.warn('logout failed:', e);
    } finally {
      setBusy(false);
      setLoggedIn(null);
      toast('Logged out');
    }
  }

  // -------------------------
  // Chat action
  // -------------------------
  async function sendChat(ev) {
    ev?.preventDefault?.();

    if (!state.user) {
      openModal();
      return;
    }

    const prompt = (chatInput?.value || '').trim();
    if (!prompt) return;

    // clear input early
    if (chatInput) chatInput.value = '';
    appendMessage('user', prompt);

    try {
      setBusy(true);
      const out = await api('/api/chat', {
        method: 'POST',
        body: { prompt, session_id: state.sessionId || null },
      });

      if (out && out.session_id) state.sessionId = out.session_id;
      appendMessage('ace', out?.response ?? '');
    } catch (e) {
      appendMessage('ace', `[ERROR] ${e.message}`);
      toast(`Chat failed: ${e.message}`, 'error');

      // If auth expired, force refresh
      if (e.status === 401) {
        await refreshMe();
      }
    } finally {
      setBusy(false);
    }
  }

  // -------------------------
  // Bind events (after DOM is ready)
  // -------------------------
  function bind() {
    // Modal open/close
    if (openAuthBtn) openAuthBtn.addEventListener('click', (e) => { e.preventDefault(); openModal(); });
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', (e) => { e.preventDefault(); closeModal(); });
    if (modalBackdrop) modalBackdrop.addEventListener('click', (e) => { e.preventDefault(); closeModal(); });

    // Tabs
    if (tabLoginBtn) tabLoginBtn.addEventListener('click', (e) => { e.preventDefault(); selectTab('login'); });
    if (tabSignupBtn) tabSignupBtn.addEventListener('click', (e) => { e.preventDefault(); selectTab('signup'); });

    // Forms
    if (loginForm) loginForm.addEventListener('submit', doLogin);
    if (signupForm) signupForm.addEventListener('submit', doSignup);

    // Some UIs use buttons instead of submit
    const loginSubmit = $('#loginSubmit') || $('[data-action="login"]');
    const signupSubmit = $('#signupSubmit') || $('[data-action="signup"]');
    if (loginSubmit) loginSubmit.addEventListener('click', doLogin);
    if (signupSubmit) signupSubmit.addEventListener('click', doSignup);

    // Logout
    if (logoutBtn) logoutBtn.addEventListener('click', doLogout);

    // Chat
    if (chatForm) chatForm.addEventListener('submit', sendChat);
    if (sendBtn) sendBtn.addEventListener('click', sendChat);

    // Enter-to-send in textarea (Shift+Enter for newline)
    if (chatInput) {
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendChat(e);
        }
      });
    }

    // ESC closes modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') closeModal();
    });

    // Initial state
    setLoggedIn(null);
  }

  async function boot() {
    bind();

    // Give server a moment if container just started
    for (let i = 0; i < 3; i++) {
      try {
        await api('/api/health');
        break;
      } catch {
        await sleep(250);
      }
    }

    await refreshMe();
  }

  // Run
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
