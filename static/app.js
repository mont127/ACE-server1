(() => {
  'use strict';

  // ===== Helpers =====
  const $ = (id) => document.getElementById(id);

  const banner = $('banner');
  const messagesEl = $('messages');

  const state = {
    me: null,
    session_id: null,
    sending: false,
  };

  function showBanner(msg, isError = true) {
    if (!banner) return;
    banner.textContent = msg || '';
    banner.style.background = isError ? '#2a1212' : '#132a18';
    banner.style.borderColor = isError ? '#5a2222' : '#1f4b2b';
    banner.classList.toggle('show', !!msg);
  }

  function log(...args) {
    console.log('[ACE UI]', ...args);
  }

  async function api(path, { method = 'GET', body = null } = {}) {
    const opts = {
      method,
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
    };
    if (body !== null) opts.body = JSON.stringify(body);

    const res = await fetch(path, opts);
    const text = await res.text();

    let data = null;
    try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }

    if (!res.ok) {
      let detail = null;
      if (data && (data.detail !== undefined)) detail = data.detail;
      else if (data && (data.message !== undefined)) detail = data.message;
      else if (data && (data.error !== undefined)) detail = data.error;

      // FastAPI sometimes returns detail as an object
      if (detail && typeof detail === 'object') {
        try { detail = JSON.stringify(detail); } catch { detail = String(detail); }
      }
      if (!detail) detail = `HTTP ${res.status}`;
      throw new Error(String(detail));
    }
    return data;
  }

  function addMessage(role, text) {
    if (!messagesEl) return;
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  // ===== Modal controls =====
  function openModal(backdropId) {
    const el = $(backdropId);
    if (!el) return;
    el.classList.add('show');
    el.setAttribute('aria-hidden', 'false');
  }

  function closeModal(backdropId) {
    const el = $(backdropId);
    if (!el) return;
    el.classList.remove('show');
    el.setAttribute('aria-hidden', 'true');
  }

  function wireModalClose(backdropId, closeBtnId) {
    const back = $(backdropId);
    const btn = $(closeBtnId);

    if (btn) {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        closeModal(backdropId);
      });
    }

    if (back) {
      back.addEventListener('click', (e) => {
        // click outside modal closes
        if (e.target === back) closeModal(backdropId);
      });
    }
  }

  // ===== Auth UI =====
  function setAuthUI(me) {
    state.me = me;

    const authPill = $('authPill');
    const btnLogin = $('btnLogin');
    const btnSignup = $('btnSignup');
    const btnLogout = $('btnLogout');

    const ident = (me && (me.email || me.username)) ? (me.email || me.username) : '';

    if (ident) {
      if (authPill) authPill.textContent = `logged in: ${ident}`;
      if (btnLogin) btnLogin.style.display = 'none';
      if (btnSignup) btnSignup.style.display = 'none';
      if (btnLogout) btnLogout.style.display = '';
    } else {
      if (authPill) authPill.textContent = 'not logged in';
      if (btnLogin) btnLogin.style.display = '';
      if (btnSignup) btnSignup.style.display = '';
      if (btnLogout) btnLogout.style.display = 'none';
    }
  }

  async function refreshHealth() {
    try {
      const h = await api('/api/health');
      const pill = $('healthPill');
      if (pill) pill.textContent = `${h.model} | ${h.device}`;
    } catch (e) {
      const pill = $('healthPill');
      if (pill) pill.textContent = 'health error';
      log('health failed', e);
    }
  }

  async function refreshMe() {
    try {
      const me = await api('/api/auth/me');
      setAuthUI(me && me.user ? me.user : me);
      showBanner('', false);
    } catch (e) {
      setAuthUI(null);
    }
  }

  async function signup(email, password) {
    // Backend expects `username` (and may also accept `email`).
    // We send both to stay compatible with either schema.
    const out = await api('/api/auth/signup', {
      method: 'POST',
      body: { username: email, email, password },
    });
    return out;
  }

  async function login(email, password) {
    // Backend expects `username` (and may also accept `email`).
    const out = await api('/api/auth/login', {
      method: 'POST',
      body: { username: email, email, password },
    });
    return out;
  }

  async function logout() {
    try {
      await api('/api/auth/logout', { method: 'POST', body: {} });
    } catch (_) {
      // ignore
    }
  }

  // ===== Chat =====
  async function sendChat(text) {
    if (state.sending) return;
    state.sending = true;

    try {
      showBanner('', false);
      addMessage('user', text);

      const payload = { prompt: text };
      if (state.session_id) payload.session_id = state.session_id;

      const out = await api('/api/chat', { method: 'POST', body: payload });
      state.session_id = out.session_id;
      addMessage('ace', out.response);
    } catch (e) {
      addMessage('ace', `Error: ${e.message}`);
      showBanner(`Error: ${e.message}`, true);
      log('chat failed', e);
    } finally {
      state.sending = false;
    }
  }

  // ===== Wire events =====
  function wireEvents() {
    const btnLogin = $('btnLogin');
    const btnSignup = $('btnSignup');
    const btnLogout = $('btnLogout');

    if (btnLogin) btnLogin.addEventListener('click', () => openModal('loginBackdrop'));
    if (btnSignup) btnSignup.addEventListener('click', () => openModal('signupBackdrop'));
    if (btnLogout) btnLogout.addEventListener('click', async () => {
      await logout();
      setAuthUI(null);
      showBanner('Logged out', false);
    });

    wireModalClose('loginBackdrop', 'loginClose');
    wireModalClose('signupBackdrop', 'signupClose');

    const loginForm = $('loginForm');
    if (loginForm) {
      loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = $('loginEmail')?.value?.trim() || '';
        const password = $('loginPassword')?.value || '';
        try {
          await login(email, password);
          closeModal('loginBackdrop');
          await refreshMe();
          showBanner('Logged in', false);
        } catch (err) {
          showBanner(`Login failed: ${err?.message || String(err)}`, true);
        }
      });
    }

    const signupForm = $('signupForm');
    if (signupForm) {
      signupForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const email = $('signupEmail')?.value?.trim() || '';
        const password = $('signupPassword')?.value || '';
        try {
          await signup(email, password);
          // auto-login after signup
          await login(email, password);
          closeModal('signupBackdrop');
          await refreshMe();
          showBanner('Account created + logged in', false);
        } catch (err) {
          showBanner(`Signup failed: ${err?.message || String(err)}`, true);
        }
      });
    }

    const btnSend = $('btnSend');
    const chatInput = $('chatInput');

    if (btnSend && chatInput) {
      btnSend.addEventListener('click', async () => {
        const text = chatInput.value.trim();
        if (!text) return;
        chatInput.value = '';
        await sendChat(text);
      });

      chatInput.addEventListener('keydown', async (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          const text = chatInput.value.trim();
          if (!text) return;
          chatInput.value = '';
          await sendChat(text);
        }
      });
    }
  }

  // ===== Boot =====
  window.addEventListener('error', (e) => {
    showBanner(`JS error: ${e.message}`, true);
    log('window error', e);
  });

  window.addEventListener('unhandledrejection', (e) => {
    const msg = e?.reason?.message || String(e.reason || 'Promise rejection');
    showBanner(`Promise error: ${msg}`, true);
    log('unhandled rejection', e);
  });

  document.addEventListener('DOMContentLoaded', async () => {
    log('boot');
    try {
      wireEvents();
      await refreshHealth();
      await refreshMe();
      addMessage('ace', 'ACE online. Login or sign up to chat.');
    } catch (e) {


      showBanner(`Startup error: ${e.message}`, true);
      log('startup failed', e);
    }
  });
})();