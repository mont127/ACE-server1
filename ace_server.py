import os
import json
import re
import time
import hmac
import base64
import hashlib
import sqlite3
import secrets
import warnings
import threading
from collections import deque
from typing import Dict, Any, Optional, Tuple, List

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# ACE SERVER — FULL (FastAPI)
# - Auth (signup/login/logout) with SQLite
# - Cookie sessions (HttpOnly)
# - Chat requires login
# - Chat history stored in SQLite
# - Per-user filesystem memory under ./mem/
# - Queue + generation lock
# - Modes:
#     ACW      -> normal (single pass)
#     ACWFULL  -> full ACE (plan -> draft -> critique -> final)
# ======================================================

MODEL_NAME = os.getenv("ACE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Keep default sane. Full mode uses multiple passes.
MAX_NEW_TOKENS = int(os.getenv("ACE_MAX_NEW_TOKENS", "1400"))

MEM_DIR = os.getenv("ACE_MEM_DIR", "mem")
DATA_DIR = os.getenv("ACE_DATA_DIR", "data")
DB_PATH = os.getenv("ACE_DB_PATH", os.path.join(DATA_DIR, "ace.db"))

# Optional API key. If set, require X-API-Key on all API calls.
API_KEY = os.getenv("ACE_API_KEY", "")

# Cookie session settings
SESSION_COOKIE = os.getenv("ACE_SESSION_COOKIE", "ace_session")
SESSION_TTL_SECONDS = int(os.getenv("ACE_SESSION_TTL_SECONDS", "604800"))  # 7 days
COOKIE_SECURE = os.getenv("ACE_COOKIE_SECURE", "0") == "1"  # set to 1 behind HTTPS

MAX_PROMPT_CHARS = int(os.getenv("ACE_MAX_PROMPT_CHARS", "8000"))
QUEUE_MAX_WAITERS = int(os.getenv("ACE_QUEUE_MAX_WAITERS", "32"))

# Basic in-memory rate limiting (per IP)
CHAT_RPM = int(os.getenv("ACE_CHAT_RPM", "30"))            # requests per minute per IP
LOGIN_RPM = int(os.getenv("ACE_LOGIN_RPM", "12"))          # login attempts per minute per IP
SIGNUP_RPM = int(os.getenv("ACE_SIGNUP_RPM", "6"))         # signup attempts per minute per IP
FULL_CHAT_RPM = int(os.getenv("ACE_FULL_CHAT_RPM", "6"))   # ACWFULL requests per minute per IP

# Session-id hardening: derive a server-side chat session id from the cookie token.
# IMPORTANT: set this to a strong random value in production and keep it stable across restarts.
SESSION_HMAC_SECRET = os.getenv("ACE_SESSION_HMAC_SECRET", "")

# Gate ACWFULL behind an explicit flag (helps prevent compute-abuse)
ENABLE_FULL_MODE = os.getenv("ACE_ENABLE_FULL_MODE", "0") == "1"

# FULL mode settings
FULL_MAX_TOKENS_PLAN = int(os.getenv("ACE_FULL_MAX_TOKENS_PLAN", "220"))
FULL_MAX_TOKENS_DRAFT = int(os.getenv("ACE_FULL_MAX_TOKENS_DRAFT", "900"))
FULL_MAX_TOKENS_CRIT = int(os.getenv("ACE_FULL_MAX_TOKENS_CRIT", "220"))
FULL_MAX_TOKENS_FINAL = int(os.getenv("ACE_FULL_MAX_TOKENS_FINAL", "900"))
SUMMARY_EVERY_TURNS = int(os.getenv("ACE_SUMMARY_EVERY_TURNS", "8"))

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use",
)

os.makedirs(MEM_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------
# Security helpers (IP + rate limiting + session id)
# ----------------------

_rl_lock = threading.Lock()


class _RateLimiter:
    def __init__(self, rpm: int):
        self.rpm = max(1, int(rpm or 1))
        self.window_s = 60
        self.hits: Dict[str, deque] = {}

    def allow(self, key: str) -> bool:
        now = _now()
        with _rl_lock:
            dq = self.hits.get(key)
            if dq is None:
                dq = deque()
                self.hits[key] = dq

            cutoff = now - self.window_s
            while dq and dq[0] < cutoff:
                dq.popleft()

            if len(dq) >= self.rpm:
                return False

            dq.append(now)
            return True


def _get_client_ip(req: Request) -> str:
    # Cloudflare Tunnel / proxy headers
    for hdr in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
        v = req.headers.get(hdr)
        if v:
            # X-Forwarded-For may be a list
            return v.split(",")[0].strip()

    if req.client and req.client.host:
        return str(req.client.host)

    return "unknown"


def _rate_limit_or_429(req: Request, limiter: _RateLimiter, scope: str) -> None:
    ip = _get_client_ip(req)
    key = f"{scope}:{ip}"
    if not limiter.allow(key):
        raise HTTPException(status_code=429, detail="Too many requests. Slow down.")


def _server_chat_session_id_from_cookie(req: Request) -> Optional[str]:
    # Derive a stable, non-sensitive session_id from the cookie token.
    token = (req.cookies.get(SESSION_COOKIE, "") or "").strip()
    if not token:
        return None

    # If the secret is missing, fall back to a hashed token (still avoids storing raw token).
    secret = SESSION_HMAC_SECRET.strip()
    if secret:
        digest = hmac.new(secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256).hexdigest()
    else:
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()

    return digest[:24]


_chat_rl = _RateLimiter(CHAT_RPM)
_full_chat_rl = _RateLimiter(FULL_CHAT_RPM)
_login_rl = _RateLimiter(LOGIN_RPM)
_signup_rl = _RateLimiter(SIGNUP_RPM)


# ----------------------
# SQLite helpers
# ----------------------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> int:
    return int(time.time())


def init_db() -> None:
    conn = db()
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT UNIQUE,
            salt BLOB NOT NULL,
            pw_hash BLOB NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )

    # Sessions
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            expires_at INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )

    # Chat history
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            mode TEXT,
            created_at INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_time ON messages(user_id, created_at)")

    # Lightweight migrations (safe no-ops on new DBs)
    try:
        cur.execute("ALTER TABLE users ADD COLUMN email TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)")
    except Exception:
        pass

    conn.commit()
    conn.close()


# ----------------------
# Auth helpers
# ----------------------

def _pbkdf2_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)


def _consteq(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def _is_valid_email(email: str) -> bool:
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email.strip()))


def _username_from_email(email: str) -> str:
    local = email.split("@", 1)[0].strip().lower()
    local = re.sub(r"[^a-z0-9_\-\.]", "_", local)
    local = re.sub(r"_+", "_", local).strip("._-")
    return (local or "user")[:24]


def _unique_username(base: str) -> str:
    conn = db()
    cur = conn.cursor()
    candidate = base
    for _ in range(25):
        cur.execute("SELECT 1 FROM users WHERE username = ?", (candidate,))
        if not cur.fetchone():
            conn.close()
            return candidate
        candidate = f"{base}_{secrets.token_hex(3)}"[:32]
    conn.close()
    return f"{base}_{secrets.token_hex(4)}"[:32]


def create_user(username: Optional[str], email: Optional[str], password: str) -> int:
    username = (username or "").strip()
    email = (email or "").strip().lower()

    if not username and not email:
        raise HTTPException(status_code=400, detail="username or email is required")

    if email and not _is_valid_email(email):
        raise HTTPException(status_code=400, detail="invalid email")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")

    if not username:
        username = _unique_username(_username_from_email(email))

    if not (3 <= len(username) <= 32) or not re.fullmatch(r"[A-Za-z0-9_\-\.]+", username):
        raise HTTPException(status_code=400, detail="username must be 3-32 chars and only A-Z a-z 0-9 _ - .")

    salt = secrets.token_bytes(16)
    pw_hash = _pbkdf2_hash(password, salt)

    conn = db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, email, salt, pw_hash, created_at) VALUES (?, ?, ?, ?, ?)",
            (username, email if email else None, salt, pw_hash, _now()),
        )
        conn.commit()
        return int(cur.lastrowid)
    except sqlite3.IntegrityError as e:
        msg = str(e).lower()
        if "username" in msg:
            raise HTTPException(status_code=409, detail="username already exists")
        if "email" in msg:
            raise HTTPException(status_code=409, detail="email already exists")
        raise HTTPException(status_code=409, detail="user already exists")
    finally:
        conn.close()


def verify_user(identifier: str, password: str) -> Optional[int]:
    ident = (identifier or "").strip()
    if not ident:
        return None

    conn = db()
    cur = conn.cursor()

    if "@" in ident:
        cur.execute("SELECT id, salt, pw_hash FROM users WHERE email = ?", (ident.lower(),))
    else:
        cur.execute("SELECT id, salt, pw_hash FROM users WHERE username = ?", (ident,))

    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    salt = bytes(row["salt"])
    expected = bytes(row["pw_hash"])
    got = _pbkdf2_hash(password, salt)

    if _consteq(expected, got):
        return int(row["id"])
    return None


def create_session(user_id: int) -> str:
    token = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO sessions (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
        (token, user_id, _now() + SESSION_TTL_SECONDS, _now()),
    )
    conn.commit()
    conn.close()
    return token


def delete_session(token: str) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


def get_user_id_from_session(token: str) -> Optional[int]:
    if not token:
        return None
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT user_id, expires_at FROM sessions WHERE token = ?", (token,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    if int(row["expires_at"]) < _now():
        try:
            delete_session(token)
        except Exception:
            pass
        return None

    return int(row["user_id"])


# ----------------------
# Chat history (DB)
# ----------------------

def add_message(user_id: int, session_id: Optional[str], role: str, content: str, mode: Optional[str] = None) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (user_id, session_id, role, content, mode, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, session_id, role, content, mode, _now()),
    )
    conn.commit()
    conn.close()


def list_messages(user_id: int, limit: int = 200) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit or 200), 500))
    conn = db()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, session_id, role, content, mode, created_at "
        "FROM messages WHERE user_id = ? "
        "ORDER BY created_at DESC, id DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows[::-1]:  # oldest -> newest
        out.append(
            {
                "id": int(r["id"]),
                "session_id": r["session_id"],
                "role": r["role"],
                "content": r["content"],
                "mode": r["mode"],
                "created_at": int(r["created_at"]),
            }
        )
    return out


def clear_messages(user_id: int) -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()


# ----------------------
# Device + Model setup
# ----------------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


device = get_device()
print("[ACE] Loading model:", MODEL_NAME)
print("[ACE] Using device:", device)

dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=dtype)
model.to(device)
model.eval()
model.config.use_cache = True

with torch.inference_mode():
    warm_inputs = tokenizer("Warmup.", return_tensors="pt").to(device)
    _ = model.generate(
        **warm_inputs,
        do_sample=False,
        max_new_tokens=8,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

GEN_LOCK = threading.Lock()
QUEUE_SEM = threading.Semaphore(QUEUE_MAX_WAITERS)


# ----------------------
# Core generation helper
# ----------------------

def _is_qwen_model() -> bool:
    return "qwen" in (MODEL_NAME or "").lower()


def _build_qwen_chat_inputs(user_text: str, system_text: Optional[str] = None):
    messages = []
    if system_text:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})

    enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )

    if isinstance(enc, torch.Tensor):
        input_ids = enc
        attention_mask = torch.ones_like(input_ids)
    else:
        input_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
        if isinstance(enc, dict) and "attention_mask" in enc:
            attention_mask = enc["attention_mask"]
        elif hasattr(enc, "attention_mask") and enc.attention_mask is not None:
            attention_mask = enc.attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
    }


def generate_text(
    prompt: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_new_tokens: int = MAX_NEW_TOKENS,
    system_text: Optional[str] = None,
) -> str:
    try:
        if _is_qwen_model():
            inputs = _build_qwen_chat_inputs(user_text=prompt, system_text=system_text)
            in_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    do_sample=True,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=8,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = output_ids[0][in_len:]
            completion = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not completion:
                completion = tokenizer.decode(gen_ids, skip_special_tokens=False).strip()
            return completion

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                min_new_tokens=8,
                temperature=temperature,
                top_p=top_p,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        completion = full_text[len(prompt):].strip()
        return completion or full_text.strip()
    except Exception as e:
        return f"[ACE ERROR] {e}"


# ----------------------
# Modes / Parsing
# ----------------------

MODE_NORMAL = "normal"
MODE_FULL = "full"


def parse_mode_and_strip(prompt: str) -> Tuple[str, str]:
    """ACWFULL wins over ACW; strip markers."""
    p = (prompt or "").strip()
    parts = p.split()
    upper_parts = [x.upper() for x in parts]

    mode = MODE_NORMAL
    if "ACWFULL" in upper_parts or p.upper().endswith("ACWFULL"):
        mode = MODE_FULL
        parts = [x for x in parts if x.upper() != "ACWFULL"]
    elif "ACW" in upper_parts or p.upper().endswith("ACW"):
        mode = MODE_NORMAL
        parts = [x for x in parts if x.upper() != "ACW"]

    stripped = " ".join(parts).strip()
    return mode, stripped


# ----------------------
# ACE logic
# ----------------------

DEFAULT_SYSTEM_ASSISTANT = (
    "You are ACE, a helpful assistant. "
    "Follow the user's constraints precisely. "
    "No jokes, no filler, no meta commentary. "
    "Do not mention policies or safety rules unless the user explicitly asks about them."
)


def is_short_greeting(prompt: str) -> bool:
    p = prompt.strip().lower()
    if not p:
        return False
    words = p.split()
    if len(words) > 3:
        return False
    greeting_tokens = ["hi", "hello", "hey", "yo", "sup", "ahoj", "čau", "čaute", "cau"]
    return any(tok in p for tok in greeting_tokens)


def dynamic_max_tokens(prompt: str) -> int:
    words = len(prompt.strip().split())
    if words == 0:
        return MAX_NEW_TOKENS
    if words <= 3:
        return min(60, MAX_NEW_TOKENS)
    if words <= 15:
        return min(260, MAX_NEW_TOKENS)
    return MAX_NEW_TOKENS


def ace_normal(prompt: str, mem: Dict[str, Any]) -> str:
    clean_prompt = (prompt or "").strip()
    if is_short_greeting(clean_prompt):
        return "Hello, I'm ACE. What do you want to try?"

    max_tokens = dynamic_max_tokens(clean_prompt)

    base_prompt = (
        "You are ACE, a helpful assistant.\n"
        "Respond only to the user's last message.\n"
        "Do not invent extra context the user did not mention.\n"
        "Keep the answer relatively short when the question is short.\n\n"
        f"User: {clean_prompt}\n"
        "ACE:"
    )

    return generate_text(
        base_prompt,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=max_tokens,
        system_text=DEFAULT_SYSTEM_ASSISTANT,
    )


def _mem_get_summary(mem: Dict[str, Any]) -> str:
    s = mem.get("summary")
    return s.strip() if isinstance(s, str) else ""


def _mem_get_history(mem: Dict[str, Any]) -> List[Dict[str, str]]:
    h = mem.get("history")
    if isinstance(h, list):
        out = []
        for it in h:
            if isinstance(it, dict) and "role" in it and "content" in it:
                out.append({"role": str(it["role"]), "content": str(it["content"])})
        return out
    return []


def _mem_push_turn(mem: Dict[str, Any], user_text: str, assistant_text: str) -> None:
    hist = _mem_get_history(mem)
    hist.append({"role": "user", "content": user_text})
    hist.append({"role": "assistant", "content": assistant_text})
    mem["history"] = hist[-24:]
    mem["turns"] = int(mem.get("turns", 0)) + 1


def _maybe_update_summary(mem: Dict[str, Any]) -> None:
    turns = int(mem.get("turns", 0))
    if turns <= 0 or (turns % SUMMARY_EVERY_TURNS) != 0:
        return

    hist = _mem_get_history(mem)
    if not hist:
        return

    prev_summary = _mem_get_summary(mem)
    convo_text = "\n".join([f'{m["role"].upper()}: {m["content"]}' for m in hist])

    prompt = (
        "Update the running memory summary.\n"
        "Write a compact summary of stable facts and user preferences.\n"
        "Do not include private data like passwords.\n"
        "Keep it under 1200 characters.\n\n"
        f"PREVIOUS SUMMARY:\n{prev_summary}\n\n"
        f"RECENT DIALOG:\n{convo_text}\n\n"
        "NEW SUMMARY:"
    )

    new_summary = generate_text(
        prompt,
        temperature=0.2,
        top_p=0.9,
        max_new_tokens=260,
        system_text=DEFAULT_SYSTEM_ASSISTANT,
    )

    mem["summary"] = (new_summary or prev_summary).strip()


def ace_full(prompt: str, mem: Dict[str, Any]) -> str:
    clean = (prompt or "").strip()
    if not clean:
        return ""

    summary = _mem_get_summary(mem)

    plan_prompt = (
        "You are ACE FULL.\n"
        "Make a short plan to answer the user's last message.\n"
        "Plan should be bullet points.\n"
        "Do NOT include the final answer.\n\n"
        f"MEMORY SUMMARY:\n{summary}\n\n"
        f"USER MESSAGE:\n{clean}\n\n"
        "PLAN:"
    )
    plan = generate_text(plan_prompt, temperature=0.3, top_p=0.9, max_new_tokens=FULL_MAX_TOKENS_PLAN, system_text=DEFAULT_SYSTEM_ASSISTANT)

    draft_prompt = (
        "You are ACE FULL.\n"
        "Write the best possible answer to the user.\n"
        "Be direct, practical, and correct.\n"
        "No filler.\n\n"
        f"MEMORY SUMMARY:\n{summary}\n\n"
        f"PLAN:\n{plan}\n\n"
        f"USER MESSAGE:\n{clean}\n\n"
        "DRAFT ANSWER:"
    )
    draft = generate_text(draft_prompt, temperature=0.7, top_p=0.92, max_new_tokens=FULL_MAX_TOKENS_DRAFT, system_text=DEFAULT_SYSTEM_ASSISTANT)

    crit_prompt = (
        "Critique the draft for errors, missing steps, or unclear parts.\n"
        "Return only the issues as bullets.\n\n"
        f"USER MESSAGE:\n{clean}\n\n"
        f"DRAFT:\n{draft}\n\n"
        "ISSUES:"
    )
    issues = generate_text(crit_prompt, temperature=0.25, top_p=0.9, max_new_tokens=FULL_MAX_TOKENS_CRIT, system_text=DEFAULT_SYSTEM_ASSISTANT)

    final_prompt = (
        "Rewrite the answer fixing the issues.\n"
        "Return ONLY the final answer.\n"
        "No headings unless the user asked for them.\n\n"
        f"USER MESSAGE:\n{clean}\n\n"
        f"DRAFT:\n{draft}\n\n"
        f"ISSUES:\n{issues}\n\n"
        "FINAL ANSWER:"
    )
    final = generate_text(final_prompt, temperature=0.6, top_p=0.92, max_new_tokens=FULL_MAX_TOKENS_FINAL, system_text=DEFAULT_SYSTEM_ASSISTANT)

    return final.strip() or draft.strip()


# ----------------------
# Memory per user (filesystem)
# ----------------------

def _mem_path_for_user(user_id: int) -> str:
    return os.path.join(MEM_DIR, f"user_{user_id}.json")


def load_mem_for_user(user_id: int) -> Dict[str, Any]:
    path = _mem_path_for_user(user_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if isinstance(d, dict):
                    return d
        except Exception:
            pass
    return {"turns": 0, "summary": "", "history": []}


def save_mem_for_user(user_id: int, mem: Dict[str, Any]) -> None:
    path = _mem_path_for_user(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ----------------------
# FastAPI app
# ----------------------

app = FastAPI(title="ACE Server", version="FULL")
app.mount("/static", StaticFiles(directory="static"), name="static")


class SignupIn(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class LoginIn(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    password: str


class ChatIn(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class ChatOut(BaseModel):
    response: str
    device: str
    model: str
    mode: str


class HistoryOut(BaseModel):
    messages: List[Dict[str, Any]]


def _require_api_key(req: Request) -> None:
    if not API_KEY:
        return
    got = req.headers.get("X-API-Key", "")
    if got != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _require_login(req: Request) -> int:
    token = req.cookies.get(SESSION_COOKIE, "")
    uid = get_user_id_from_session(token)
    if not uid:
        raise HTTPException(status_code=401, detail="Not logged in")
    return uid


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health(request: Request):
    _require_api_key(request)
    return {
        "ok": True,
        "model": MODEL_NAME,
        "device": device,
        "queue_max_waiters": QUEUE_MAX_WAITERS,
        "modes": ["ACW", "ACWFULL"],
    }


# ---- Auth endpoints ----

@app.post("/api/auth/signup")
def signup(payload: SignupIn, request: Request):
    _require_api_key(request)
    _rate_limit_or_429(request, _signup_rl, "signup")

    uid = create_user(username=payload.username, email=payload.email, password=payload.password)

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE id = ?", (uid,))
    row = cur.fetchone()
    conn.close()

    username = row["username"] if row else (payload.username or "")

    token = create_session(uid)

    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="Lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    return resp


@app.post("/api/auth/login")
def login(payload: LoginIn, request: Request):
    _require_api_key(request)
    _rate_limit_or_429(request, _login_rl, "login")

    identifier = (payload.username or payload.email or "").strip()
    if not identifier:
        raise HTTPException(status_code=400, detail="username or email is required")

    uid = verify_user(identifier, payload.password)
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE id = ?", (uid,))
    row = cur.fetchone()
    conn.close()

    username = row["username"] if row else identifier

    token = create_session(uid)
    resp = JSONResponse({"ok": True, "username": username})
    resp.set_cookie(
        SESSION_COOKIE,
        token,
        httponly=True,
        secure=COOKIE_SECURE,
        samesite="Lax",
        max_age=SESSION_TTL_SECONDS,
        path="/",
    )
    return resp


@app.post("/api/auth/logout")
def logout(request: Request):
    _require_api_key(request)

    token = request.cookies.get(SESSION_COOKIE, "")
    if token:
        try:
            delete_session(token)
        except Exception:
            pass

    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
def me(request: Request):
    _require_api_key(request)

    token = request.cookies.get(SESSION_COOKIE, "")
    uid = get_user_id_from_session(token)
    if not uid:
        return {"logged_in": False}

    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT username FROM users WHERE id = ?", (uid,))
    row = cur.fetchone()
    conn.close()

    return {"logged_in": True, "user_id": uid, "username": row["username"] if row else ""}


# ---- History endpoints ----

@app.get("/api/chat/history", response_model=HistoryOut)
def chat_history(request: Request, limit: int = 200):
    _require_api_key(request)
    uid = _require_login(request)
    return {"messages": list_messages(uid, limit=limit)}


@app.post("/api/chat/clear")
def chat_clear(request: Request):
    _require_api_key(request)
    uid = _require_login(request)
    clear_messages(uid)
    save_mem_for_user(uid, {"turns": 0, "summary": "", "history": []})
    return {"ok": True}


# ---- Chat ----

@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn, request: Request):
    _require_api_key(request)
    uid = _require_login(request)
    # Rate limit per IP
    _rate_limit_or_429(request, _chat_rl, "chat")

    raw_prompt = (payload.prompt or "").strip()
    if not raw_prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(raw_prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"prompt too large (max {MAX_PROMPT_CHARS} chars)")

    mode, prompt = parse_mode_and_strip(raw_prompt)
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    # Gate ACWFULL to reduce compute-abuse (enable with ACE_ENABLE_FULL_MODE=1)
    if mode == MODE_FULL and not ENABLE_FULL_MODE:
        raise HTTPException(status_code=403, detail="ACWFULL is disabled")

    # Additional tighter rate limit for ACWFULL
    if mode == MODE_FULL:
        _rate_limit_or_429(request, _full_chat_rl, "full")

    # Ignore client-provided session_id to prevent spoofing; derive server-side session id.
    sid = _server_chat_session_id_from_cookie(request)

    if not QUEUE_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy. Try again in a moment.")

    try:
        mem = load_mem_for_user(uid)

        with GEN_LOCK:
            if mode == MODE_FULL:
                resp_text = ace_full(prompt, mem)
            else:
                resp_text = ace_normal(prompt, mem)

        # Persist chat history
        add_message(uid, sid, "user", prompt, mode=mode)
        add_message(uid, sid, "assistant", resp_text, mode=mode)

        # Update memory
        _mem_push_turn(mem, prompt, resp_text)
        _maybe_update_summary(mem)
        save_mem_for_user(uid, mem)

        return ChatOut(response=resp_text, device=device, model=MODEL_NAME, mode=mode)

    finally:
        QUEUE_SEM.release()