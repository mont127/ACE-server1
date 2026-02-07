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
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# ACE v7.3 — SERVER EDITION (FastAPI)
# + Auth (signup/login) with SQLite
# + Cookie sessions (HttpOnly)
# + Bounded queue + generation lock
# + Per-user filesystem memory under ./mem/
# ======================================================

MODEL_NAME = os.getenv("ACE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("ACE_MAX_NEW_TOKENS", "7000"))
MEM_DIR = os.getenv("ACE_MEM_DIR", "mem")
DATA_DIR = os.getenv("ACE_DATA_DIR", "data")
DB_PATH = os.getenv("ACE_DB_PATH", os.path.join(DATA_DIR, "ace.db"))

# Optional API key. If set, require X-API-Key on all API calls.
API_KEY = os.getenv("ACE_API_KEY", "")

# Auth/session settings
SESSION_COOKIE = os.getenv("ACE_SESSION_COOKIE", "ace_session")
SESSION_TTL_SECONDS = int(os.getenv("ACE_SESSION_TTL_SECONDS", "604800"))  # 7 days
COOKIE_SECURE = os.getenv("ACE_COOKIE_SECURE", "0") == "1"  # set to 1 behind HTTPS

# Safety/abuse limits
MAX_PROMPT_CHARS = int(os.getenv("ACE_MAX_PROMPT_CHARS", "8000"))
QUEUE_MAX_WAITERS = int(os.getenv("ACE_QUEUE_MAX_WAITERS", "32"))

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use",
)

os.makedirs(MEM_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------
# SQLite helpers
# ----------------------

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = db()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            salt BLOB NOT NULL,
            pw_hash BLOB NOT NULL,
            created_at INTEGER NOT NULL
        );
        """
    )
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
    conn.commit()
    conn.close()


def _now() -> int:
    return int(time.time())


def _pbkdf2_hash(password: str, salt: bytes) -> bytes:
    # PBKDF2-HMAC-SHA256
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 210_000)


def _consteq(a: bytes, b: bytes) -> bool:
    return hmac.compare_digest(a, b)


def create_user(username: str, password: str) -> int:
    username = username.strip()
    if not (3 <= len(username) <= 32) or not re.fullmatch(r"[A-Za-z0-9_\-\.]+", username):
        raise HTTPException(status_code=400, detail="username must be 3-32 chars and only A-Z a-z 0-9 _ - .")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="password must be at least 8 characters")

    salt = secrets.token_bytes(16)
    pw_hash = _pbkdf2_hash(password, salt)

    conn = db()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, salt, pw_hash, created_at) VALUES (?, ?, ?, ?)",
            (username, salt, pw_hash, _now()),
        )
        conn.commit()
        user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="username already exists")
    finally:
        conn.close()

    return user_id


def verify_user(username: str, password: str) -> Optional[int]:
    conn = db()
    cur = conn.cursor()
    cur.execute("SELECT id, salt, pw_hash FROM users WHERE username = ?", (username.strip(),))
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
        # expired
        try:
            delete_session(token)
        except Exception:
            pass
        return None
    return int(row["user_id"])


# ----------------------
# Device + Model Setup
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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
)
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

# Serialize actual generation (especially on CPU)
GEN_LOCK = threading.Lock()

# Bounded waiting room (prevents infinite pile-up)
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
# ACE logic (minimal, keep yours if you want)
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
        return min(40, MAX_NEW_TOKENS)
    if words <= 15:
        return min(180, MAX_NEW_TOKENS)
    return MAX_NEW_TOKENS


def ace_once(prompt: str, mem: Dict[str, Any]) -> str:
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
                return json.load(f)
        except Exception:
            pass
    return {"scores": [], "last_story": ""}


def save_mem_for_user(user_id: int, mem: Dict[str, Any]) -> None:
    path = _mem_path_for_user(user_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mem, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ----------------------
# FastAPI App + UI
# ----------------------

app = FastAPI(title="ACE Server", version="7.3")
app.mount("/static", StaticFiles(directory="static"), name="static")


class SignupIn(BaseModel):
    username: str
    password: str


class LoginIn(BaseModel):
    username: str
    password: str


class ChatIn(BaseModel):
    prompt: str


class ChatOut(BaseModel):
    response: str
    device: str
    model: str


def _require_api_key(req: Request) -> None:
    if not API_KEY:
        return
    got = req.headers.get("X-API-Key", "")
    if got != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _require_login(req: Request) -> int:
    token = req.cookies.get(SESSION_COOKIE, "")
    user_id = get_user_id_from_session(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not logged in")
    return user_id


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
    return {"ok": True, "model": MODEL_NAME, "device": device, "queue_max_waiters": QUEUE_MAX_WAITERS}


@app.post("/api/auth/signup")
def signup(payload: SignupIn, request: Request):
    _require_api_key(request)
    uid = create_user(payload.username, payload.password)
    token = create_session(uid)

    resp = JSONResponse({"ok": True, "username": payload.username.strip()})
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
    uid = verify_user(payload.username, payload.password)
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_session(uid)
    resp = JSONResponse({"ok": True, "username": payload.username.strip()})
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


@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn, request: Request):
    _require_api_key(request)

    user_id = _require_login(request)

    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(status_code=413, detail=f"prompt too large (max {MAX_PROMPT_CHARS} chars)")

    # Bounded waiting room: if too many people are waiting, reject fast.
    if not QUEUE_SEM.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Server is busy. Try again in a moment.")

    try:
        mem = load_mem_for_user(user_id)

        # Serialize actual generation.
        with GEN_LOCK:
            resp_text = ace_once(prompt, mem)

        save_mem_for_user(user_id, mem)

        return ChatOut(response=resp_text, device=device, model=MODEL_NAME)

    finally:
        QUEUE_SEM.release()