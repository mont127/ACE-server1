import os
import json
import random
import re
import warnings
import threading
from typing import Dict, Any, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# ACE v7.3 — SERVER EDITION (FastAPI)
# - Loads the model once at startup
# - Serves a simple web UI from /static and index at /
# - POST /api/chat for chat requests
# - Session memory persisted under ./mem/
# - Serializes generation with a lock (recommended on CPU)
# ======================================================

MODEL_NAME = os.getenv("ACE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_NEW_TOKENS = int(os.getenv("ACE_MAX_NEW_TOKENS", "7000"))
MEM_DIR = os.getenv("ACE_MEM_DIR", "mem")
API_KEY = os.getenv("ACE_API_KEY", "")  # optional: set to require X-API-Key

warnings.filterwarnings(
    "ignore",
    message="To copy construct from a tensor, it is recommended to use",
)

os.makedirs(MEM_DIR, exist_ok=True)

# ----------------------
# Device + Model Setup
# ----------------------

def get_device() -> str:
    # Rocky Linux server: usually CPU only.
    # If you later run CUDA, this will pick it up.
    if torch.cuda.is_available():
        return "cuda"
    # MPS is macOS only; keep for dev machines
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

# warm-up
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
# ACE logic (keep yours)
# ----------------------

# NOTE:
# This server includes the essential parts of your ACE logic.
# If you have extra helper functions in your GUI script, paste them here.

DEFAULT_SYSTEM_ASSISTANT = (
    "You are ACE, a helpful assistant. "
    "Follow the user's constraints precisely. "
    "No jokes, no filler, no meta commentary. "
    "Do not mention policies or safety rules unless the user explicitly asks about them."
)

DEFAULT_SYSTEM_STORY = (
    "You are ACE, a narrative engine. "
    "Follow the user's constraints exactly. "
    "When the user asks for multiple parts, clearly separate them with labels like 'SCP SECTION' and 'EMOTIONAL SCENE'. "
    "Do not mention policies, safety rules, or meta commentary; stay in-character."
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


def split_acw_marker(prompt: str) -> tuple[str, bool]:
    p = prompt.rstrip()
    if re.search(r"\s+acw$", p, flags=re.IGNORECASE):
        clean = re.sub(r"\s+acw$", "", p, flags=re.IGNORECASE).rstrip()
        return clean, True
    return prompt, False


def dynamic_max_tokens(prompt: str) -> int:
    words = len(prompt.strip().split())
    if words == 0:
        return MAX_NEW_TOKENS
    if words <= 3:
        return min(40, MAX_NEW_TOKENS)
    if words <= 15:
        return min(140, MAX_NEW_TOKENS)
    return MAX_NEW_TOKENS


def ace_once(prompt: str, mem: Dict[str, Any]) -> str:
    clean_prompt, acw_enabled = split_acw_marker(prompt)

    # If ACW is NOT requested, run base assistant mode.
    if not acw_enabled:
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
            temperature=0.6,
            top_p=0.9,
            max_new_tokens=max_tokens,
            system_text=DEFAULT_SYSTEM_ASSISTANT,
        )

    # ACW requested → keep your full ACW/ACC logic here.
    # For now, we use story-system to match your behavior.
    return generate_text(
        clean_prompt,
        temperature=0.95,
        top_p=0.95,
        max_new_tokens=MAX_NEW_TOKENS,
        system_text=DEFAULT_SYSTEM_STORY,
    )


# ----------------------
# Memory per session
# ----------------------

def _mem_path(session_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)
    return os.path.join(MEM_DIR, f"session_{safe}.json")


def load_mem(session_id: str) -> Dict[str, Any]:
    path = _mem_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"scores": [], "last_story": ""}


def save_mem(session_id: str, mem: Dict[str, Any]) -> None:
    path = _mem_path(session_id)
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


class ChatIn(BaseModel):
    prompt: str
    session_id: Optional[str] = None


class ChatOut(BaseModel):
    session_id: str
    response: str
    device: str
    model: str


def _require_api_key(req: Request) -> None:
    if not API_KEY:
        return
    got = req.headers.get("X-API-Key", "")
    if got != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join("static", "index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_NAME, "device": device}


@app.post("/api/chat", response_model=ChatOut)
def chat(payload: ChatIn, request: Request):
    _require_api_key(request)

    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    session_id = payload.session_id or request.cookies.get("ace_session") or os.urandom(12).hex()
    mem = load_mem(session_id)

    # Serialize generation to avoid OOM/CPU thrash.
    with GEN_LOCK:
        resp = ace_once(prompt, mem)

    save_mem(session_id, mem)

    return ChatOut(session_id=session_id, response=resp, device=device, model=MODEL_NAME)