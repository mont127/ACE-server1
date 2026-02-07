let sessionId = localStorage.getItem("ace_session_id") || "";

const chat = document.getElementById("chat");
const promptEl = document.getElementById("prompt");
const sendBtn = document.getElementById("send");
const clearBtn = document.getElementById("clear");
const statusEl = document.getElementById("status");

function addMsg(role, text) {
  const wrap = document.createElement("div");
  wrap.className = `msg ${role}`;
  const r = document.createElement("div");
  r.className = "role";
  r.textContent = role === "user" ? "USER" : "ACE";
  const b = document.createElement("div");
  b.className = "body";
  b.textContent = text;
  wrap.appendChild(r);
  wrap.appendChild(b);
  chat.appendChild(wrap);
  chat.scrollTop = chat.scrollHeight;
}

async function health() {
  try {
    const r = await fetch("/api/health");
    const j = await r.json();
    statusEl.textContent = `Model: ${j.model} | Device: ${j.device}`;
  } catch {
    statusEl.textContent = "Offline";
  }
}

async function send() {
  const text = (promptEl.value || "").trim();
  if (!text) return;

  addMsg("user", text);
  promptEl.value = "";

  sendBtn.disabled = true;
  sendBtn.textContent = "Working…";

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: text, session_id: sessionId || null })
    });

    if (!r.ok) {
      const err = await r.text();
      addMsg("ace", `Error: ${err}`);
      return;
    }

    const j = await r.json();
    sessionId = j.session_id;
    localStorage.setItem("ace_session_id", sessionId);
    addMsg("ace", j.response);

  } catch (e) {
    addMsg("ace", `Network error: ${String(e)}`);
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "Send";
  }
}

sendBtn.addEventListener("click", send);
clearBtn.addEventListener("click", () => {
  chat.innerHTML = "";
});

promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

health();