"""
brain.py — JARVIS Cognitive Architecture (Cloud API Edition)
==============================================================
AI Control Layer with intelligent query classification and model routing.
All models are 100% free — NVIDIA NIM + OpenRouter free tier.
No local GPU or Ollama required.

Models (6-tier cloud architecture):
    reflex      — deepseek-v3 (NVIDIA NIM, instant)
    analyst     — deepseek-v3 (NVIDIA NIM, reasoning)
    coder       — qwen3-coder-480b (OpenRouter :free, code)
    oracle      — deepseek-r1 (OpenRouter :free, deep reasoning)
    ultra       — llama-3.1-70b (NVIDIA NIM, large context)
    backup      — llama-3.2-3b-instruct (OpenRouter :free, always available)

Usage:
    from brain import think, think_stream, set_mode, get_mode
    result = think("Hello JARVIS")
"""

import os
import sys
import json
import logging
import time
import threading

from dotenv import load_dotenv
from openai import OpenAI

# ── Load environment variables ──────────────────────────────────────────────
load_dotenv()

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("jarvis.brain")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")

NVIDIA_BASE_URL     = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODELS = {

   "reflex": {
    "provider":    "nvidia",
    "model":       "meta/llama-3.2-3b-instruct",  # reliable, fast on NVIDIA
    "max_tokens":  256,
    "temperature": 0.7,
},

    "analyst": {
        "provider":    "nvidia",
        "model":       "deepseek-ai/deepseek-v4-flash",
        "use_for":     "explain, summarize, plan, research",
        "max_tokens":  2048,
        "temperature": 0.6,
    },

    "coder": {
        "provider":    "openrouter",
        "model":       "qwen/qwen3-coder-480b:free",
        "use_for":     "code, debug, build, implement",
        "max_tokens":  4096,
        "temperature": 0.3,
    },

    "oracle": {
        "provider":    "nvidia",
        "model":       "deepseek-ai/deepseek-r1",
        "use_for":     "complex tasks, deep reasoning, strategy",
        "max_tokens":  4096,
        "temperature": 0.4,
    },

    "ultra": {
        "provider":    "nvidia",
        "model":       "meta/llama-3.1-70b-instruct",
        "use_for":     "very long documents, large context",
        "max_tokens":  8192,
        "temperature": 0.5,
    },

    "backup": {
        "provider":    "openrouter",
        "model":       "meta-llama/llama-3.2-3b-instruct:free",
        "use_for":     "fallback when all else fails",
        "max_tokens":  1024,
        "temperature": 0.7,
    },
    # ADD these two new entries to MODELS:
    "backup2": {
    "provider":    "openrouter",
    "model":       "google/gemma-3-4b-it:free",
    "use_for":     "second fallback",
    "max_tokens":  512,
    "temperature": 0.7,
},
    "backup3": {
    "provider":    "openrouter",
    "model":       "mistralai/mistral-small-3.1:free",
    "use_for":     "third fallback",
    "max_tokens":  512,
    "temperature": 0.7,
},
}
# THEN update ALL fallback chains to include backup2 + backup3:
FALLBACK_CHAINS = {
    "reflex":  ["reflex",  "analyst", "backup", "backup2", "backup3"],
    "analyst": ["analyst", "reflex",  "backup", "backup2", "backup3"],
    "coder":   ["coder",   "oracle",  "analyst", "backup", "backup2", "backup3"],
    "oracle":  ["oracle",  "analyst", "backup", "backup2", "backup3"],
    "ultra":   ["ultra",   "oracle",  "analyst", "backup", "backup2", "backup3"],
    "backup":  ["backup",  "backup2", "backup3"],
}



FALLBACK_CHAINS = {
    "reflex":  ["reflex",  "analyst", "backup"],
    "analyst": ["analyst", "reflex",  "backup"],
    "coder":   ["coder",   "oracle",  "analyst", "backup"],
    "oracle":  ["oracle",  "analyst", "backup"],
    "ultra":   ["ultra",   "oracle",  "analyst", "backup"],
    "backup":  ["backup"],
}

CURRENT_MODE = "auto"

_is_generating = False
_gen_lock = threading.Lock()

_session_stats = {
    "reflex": 0, "analyst": 0, "coder": 0, "oracle": 0,
    "ultra": 0, "backup": 0, "total": 0, "fallbacks": 0,
}
_stats_lock = threading.Lock()

_memory = None

def _get_memory():
    global _memory
    if _memory is None:
        try:
            from memory import get_memory
            _memory = get_memory()
        except Exception as e:
            log.warning(f"[Brain] Memory system unavailable: {e}")
    return _memory


# ═══════════════════════════════════════════════════════════════════════════════
#  QUERY CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

_REFLEX_KEYWORDS = [
    "hello", "hi", "hey", "what's up", "status", "how are",
    "open ", "close ", "play ", "pause", "stop", "yes", "no",
    "time", "date", "weather", "turn on", "turn off", "volume",
    "good morning", "good evening", "thanks", "thank you",
    "ok", "sure", "bye", "goodbye", "who are you", "your name",
    "sup", "yo",
]
_CODER_KEYWORDS = [
    "code", "function", "debug", "error", "build", "implement",
    "python", "javascript", "html", "css", "api", "class", "fix",
    "script", "program", "write a", "create a function", "bug",
    "syntax", "import", "library", "framework", "deploy",
    "variable", "compile", "refactor", "algorithm", "endpoint",
    "database", "sql", "java", "react", "git", "commit",
    "docker", "test", "unit test", "regex",
]
_ORACLE_KEYWORDS = [
    "analyze", "strategy", "plan", "complex", "think through",
    "explain why", "compare", "best approach", "architecture",
    "design", "optimize", "should i", "what is the best way",
    "pros and cons", "evaluate", "review my",
]
_ULTRA_KEYWORDS = [
    "read this document", "summarize this file", "long document",
    "entire codebase", "whole project", "everything about",
]
_ANALYST_KEYWORDS = [
    "explain", "summarize", "what is", "how does", "tell me",
    "research", "find", "search", "who is", "when", "why",
    "describe", "elaborate", "step by step", "difference between",
    "recommend", "suggest", "opinion",
]


def classify_query(query: str) -> str:
    """Zero-cost keyword classifier. No API call needed."""
    q = query.lower().strip()

    if len(q) < 20 or any(kw in q for kw in _REFLEX_KEYWORDS):
        if not any(kw in q for kw in _CODER_KEYWORDS + _ANALYST_KEYWORDS + _ORACLE_KEYWORDS):
            log.info("[JARVIS CORE] → Classified: reflex | Reason: keyword_match")
            return "reflex"

    for p in _ULTRA_KEYWORDS:
        if p in q:
            log.info("[JARVIS CORE] → Classified: ultra | Reason: ultra_keyword")
            return "ultra"

    if any(kw in q for kw in _CODER_KEYWORDS):
        log.info("[JARVIS CORE] → Classified: coder | Reason: code_keyword")
        return "coder"

    for p in _ORACLE_KEYWORDS:
        if p in q:
            log.info("[JARVIS CORE] → Classified: oracle | Reason: oracle_keyword")
            return "oracle"

    if any(kw in q for kw in _ANALYST_KEYWORDS) or len(query) > 100:
        log.info("[JARVIS CORE] → Classified: analyst | Reason: analysis_keyword")
        return "analyst"

    log.info("[JARVIS CORE] → Classified: analyst | Reason: default")
    return "analyst"


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED MODEL CALLER
# ═══════════════════════════════════════════════════════════════════════════════

def _make_client(provider: str) -> OpenAI | None:
    """
    Build an OpenAI-compatible client for the given provider.
    max_retries=0 prevents the SDK from hanging on 429 errors.
    timeout=15.0 ensures requests never freeze forever.
    Returns None if the required API key is missing.
    """
    if provider == "nvidia":
        if not NVIDIA_API_KEY:
            log.warning("[JARVIS] NVIDIA key missing")
            return None
        return OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            max_retries=0,   # ← CRITICAL: stop SDK auto-retry on 429
            timeout=15.0,    # ← CRITICAL: hard timeout, never freeze forever
        )
    else:  # openrouter
        if not OPENROUTER_KEY:
            log.warning("[JARVIS] OpenRouter key missing")
            return None
        return OpenAI(
            api_key=OPENROUTER_KEY,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,   # ← CRITICAL: stop SDK auto-retry on 429
            timeout=20.0,    # OpenRouter is slightly slower than NVIDIA
        )


def _call_model_non_stream(tier: str, messages: list) -> str | None:
    """
    Non-streaming model caller.
    Returns full response text or None on failure.
    """
    cfg = MODELS.get(tier, MODELS["backup"])
    client = _make_client(cfg["provider"])

    if client is None:
        # Key missing — try backup directly
        if tier != "backup":
            return _call_model_non_stream("backup", messages)
        return "API keys not configured, sir. Please set NVIDIA_API_KEY or OPENROUTER_API_KEY in .env"

    log.info(f"[JARVIS CORE] → Tier: {tier} | Model: {cfg['model']} | Provider: {cfg['provider']}")

    try:
        response = client.chat.completions.create(
            model       = cfg["model"],
            messages    = messages,
            max_tokens  = cfg["max_tokens"],
            temperature = cfg.get("temperature", 0.7),
            stream      = False,
        )
        return response.choices[0].message.content

    except Exception as e:
        # ── FIX: err is now correctly inside the except block ──────────
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            log.warning(f"[JARVIS] {tier} rate limited — cascading immediately")
        else:
            log.error(f"[JARVIS] {tier} failed: {err}")
        return None


def _call_model_stream(tier: str, messages: list):
    """
    Streaming model caller. Yields token strings as they arrive.
    Falls back to backup tier on failure or rate limit.
    """
    cfg = MODELS.get(tier, MODELS["backup"])
    client = _make_client(cfg["provider"])

    if client is None:
        if tier != "backup":
            yield from _call_model_stream("backup", messages)
        else:
            yield "API keys not configured, sir. Please set NVIDIA_API_KEY or OPENROUTER_API_KEY in .env"
        return

    log.info(
        f"[JARVIS CORE] → Tier: {tier} | Model: {cfg['model']} "
        f"| Provider: {cfg['provider']} | Stream: True"
    )

    try:
        response = client.chat.completions.create(
            model       = cfg["model"],
            messages    = messages,
            max_tokens  = cfg["max_tokens"],
            temperature = cfg.get("temperature", 0.7),
            stream      = True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        # ── FIX: err is now correctly inside the except block ──────────
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            log.warning(f"[JARVIS] {tier} rate limited — cascading immediately (no retry wait)")
        else:
            log.error(f"[JARVIS] {tier} stream failed: {err}")

        if tier != "backup":
            log.info(f"[JARVIS] Cascading to backup tier...")
            yield from _call_model_stream("backup", messages)
        else:
            yield "All providers are currently unavailable, sir. Try again in a moment."


# ═══════════════════════════════════════════════════════════════════════════════
#  MODE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def set_mode(mode: str) -> bool:
    global CURRENT_MODE
    mode = mode.strip().lower()
    if mode not in ("fast", "smart", "auto"):
        return False
    CURRENT_MODE = mode
    log.info(f"[AI] Mode set to: {CURRENT_MODE.upper()}")
    return True


def get_mode() -> dict:
    return {
        "mode": CURRENT_MODE,
        "models": {k: v["model"] for k, v in MODELS.items()},
        "active_model": _resolve_model_name(),
    }


def _resolve_model_name() -> str:
    if CURRENT_MODE == "fast":
        return MODELS["reflex"]["model"]
    elif CURRENT_MODE == "smart":
        return MODELS["oracle"]["model"]
    return "auto"


def get_session_stats() -> dict:
    with _stats_lock:
        return dict(_session_stats)


def _record_stat(tier: str, is_fallback: bool = False):
    with _stats_lock:
        _session_stats["total"] += 1
        if tier in _session_stats:
            _session_stats[tier] += 1
        if is_fallback:
            _session_stats["fallbacks"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def think(prompt: str) -> dict:
    """
    Full JARVIS cognitive pipeline (non-streaming).
    1. Memory retrieval
    2. System prompt assembly
    3. Query classification
    4. Model routing with fallback chain
    5. Response generation
    6. Memory update
    """
    mem = _get_memory()
    memory_used = False
    retrieved_episodes = []

    if mem:
        try:
            retrieved_episodes = mem.retrieve(prompt, top_k=3)
            memory_used = len(retrieved_episodes) > 0
        except Exception as e:
            log.warning(f"[Brain] Memory retrieval failed: {e}")

    if mem:
        system_prompt = mem.build_system_prompt(retrieved_episodes)
    else:
        from memory import JARVIS_PERSONA
        system_prompt = JARVIS_PERSONA

    if mem:
        procedure = mem.match_procedure(prompt)
        if procedure:
            steps_str = ", ".join(procedure.get("steps", []))
            prompt = (
                f"{prompt}\n\n[JARVIS INTERNAL: User triggered procedure "
                f"'{procedure['name']}'. Steps: {steps_str}. "
                f"Acknowledge and describe executing these steps.]"
            )

        remember_cmd = mem.detect_remember_command(prompt)
        if remember_cmd:
            mem.add_procedure(
                remember_cmd["name"],
                remember_cmd["steps"],
                triggers=[remember_cmd["name"].replace("_", " ")]
            )
            response = (
                f"Understood, sir. I've memorized the procedure "
                f"'{remember_cmd['name']}' with {len(remember_cmd['steps'])} steps: "
                f"{', '.join(remember_cmd['steps'])}. "
                f"Just say '{remember_cmd['name'].replace('_', ' ')}' and I'll execute it."
            )
            mem.remember(prompt, response, tags=["procedure", "memory"])
            return {
                "action": "NONE", "response": response,
                "model": "internal", "fallback": False,
                "tier": "internal", "memory_used": False,
            }

    t_start = time.time()

    if CURRENT_MODE == "auto":
        tier = classify_query(prompt)
    elif CURRENT_MODE == "fast":
        tier = "reflex"
    elif CURRENT_MODE == "smart":
        tier = "oracle"
    else:
        tier = "analyst"

    chain = FALLBACK_CHAINS.get(tier, ["backup"])
    fallback_used = False
    response = None
    used_model = "none"
    used_tier = tier

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    for i, model_key in enumerate(chain):
        model_info = MODELS.get(model_key)
        if not model_info:
            continue
        if i > 0:
            fallback_used = True
            log.warning(f"[JARVIS CORE] Fallback → {model_key}")

        log.info(
            f"[JARVIS CORE] → Routing to: {model_key} "
            f"| Model: {model_info['model']} | Provider: {model_info['provider']} | Free: True"
        )

        response = _call_model_non_stream(model_key, messages)
        if response:
            used_model = model_info["model"]
            used_tier = model_key
            _record_stat(model_key, is_fallback=fallback_used)
            break

    if not response:
        response = (
            "All AI engines are currently offline, sir. "
            "Please check NVIDIA_API_KEY and OPENROUTER_API_KEY in your .env file."
        )
        used_model = "none"
        fallback_used = True

    if mem:
        try:
            mem.remember(prompt, response)
            mem.update_semantic_from_message(prompt)
        except Exception as e:
            log.warning(f"[Brain] Memory update failed: {e}")

    t_elapsed = time.time() - t_start
    log.info(
        f"[PERF] think() | Model: {used_model} | Tier: {used_tier} "
        f"| Time: {t_elapsed:.2f}s | Fallback: {fallback_used} | Memory: {memory_used}"
    )

    return {
        "action": "NONE",
        "response": response,
        "model": used_model,
        "fallback": fallback_used,
        "tier": used_tier,
        "memory_used": memory_used,
        "response_time_ms": int(t_elapsed * 1000),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMING ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def think_stream(prompt: str):
    """
    Generator — yields tokens as they arrive for SSE streaming in server.py.
    Yields {"token": str} per token, {"done": True, ...} at end.
    """
    mem = _get_memory()
    retrieved_episodes = []
    memory_used = False

    if mem:
        try:
            retrieved_episodes = mem.retrieve(prompt, top_k=3)
            memory_used = len(retrieved_episodes) > 0
        except Exception:
            pass

    if mem:
        system_prompt = mem.build_system_prompt(retrieved_episodes)
    else:
        from memory import JARVIS_PERSONA
        system_prompt = JARVIS_PERSONA

    if mem:
        remember_cmd = mem.detect_remember_command(prompt)
        if remember_cmd:
            mem.add_procedure(
                remember_cmd["name"], remember_cmd["steps"],
                triggers=[remember_cmd["name"].replace("_", " ")]
            )
            response = (
                f"Understood, sir. I've memorized '{remember_cmd['name']}' "
                f"with {len(remember_cmd['steps'])} steps."
            )
            for ch in response:
                yield {"token": ch}
            yield {"done": True, "model": "internal", "tier": "internal",
                   "fallback": False, "memory_used": False}
            return

    t_start = time.time()

    if CURRENT_MODE == "auto":
        tier = classify_query(prompt)
    elif CURRENT_MODE == "fast":
        tier = "reflex"
    elif CURRENT_MODE == "smart":
        tier = "oracle"
    else:
        tier = "analyst"

    chain = FALLBACK_CHAINS.get(tier, ["backup"])
    fallback_used = False
    full_response = ""
    first_token_time = None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": prompt},
    ]

    for i, model_key in enumerate(chain):
        model_info = MODELS.get(model_key)
        if not model_info:
            continue
        if i > 0:
            fallback_used = True

        log.info(
            f"[JARVIS CORE] → Streaming from: {model_key} "
            f"| {model_info['model']} | {model_info['provider']}"
        )

        try:
            streamed = False

            with _gen_lock:
                global _is_generating
                _is_generating = True

            try:
                for token in _call_model_stream(model_key, messages):
                    if first_token_time is None:
                        first_token_time = time.time()
                        log.info(
                            f"[PERF] First token latency: "
                            f"{(first_token_time - t_start)*1000:.0f}ms"
                        )
                    full_response += token
                    yield {"token": token}
                    streamed = True
            finally:
                with _gen_lock:
                    _is_generating = False

            if streamed:
                t_elapsed = time.time() - t_start
                log.info(
                    f"[PERF] stream() | Model: {model_info['model']} "
                    f"| Time: {t_elapsed:.2f}s | Chars: {len(full_response)}"
                )
                _record_stat(model_key, fallback_used)
                if mem:
                    try:
                        mem.remember(prompt, full_response)
                        mem.update_semantic_from_message(prompt)
                    except Exception:
                        pass
                yield {
                    "done": True,
                    "model": model_info["model"],
                    "tier": model_key,
                    "fallback": fallback_used,
                    "memory_used": memory_used,
                    "response_time_ms": int(t_elapsed * 1000),
                }
                return

        except Exception as e:
            # ── FIX: err is now correctly inside the except block ──────
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                log.warning(
                    f"[JARVIS] {model_key} rate limited — "
                    f"trying next in chain (no retry wait)"
                )
            else:
                log.error(f"[JARVIS] {model_key} stream exception: {err}")
            # Continue to next model in chain — don't stop here
            continue

    # ── All models in chain exhausted ───────────────────────────────────
    error_msg = "All engines offline, sir. Check your API keys in the .env file."
    for ch in error_msg:
        yield {"token": ch}
    yield {"done": True, "model": "none", "tier": "none",
           "fallback": True, "memory_used": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_nvidia() -> bool:
    return bool(NVIDIA_API_KEY)

def check_openrouter() -> bool:
    return bool(OPENROUTER_KEY)

def check_ollama() -> bool:
    """Backward-compatible — checks if any cloud API key is configured."""
    return check_nvidia() or check_openrouter()

def check_gemini() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPAT ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

def gemini_think(prompt: str) -> str | None:
    from memory import JARVIS_PERSONA
    messages = [{"role": "system", "content": JARVIS_PERSONA},
                {"role": "user",   "content": prompt}]
    return _call_model_non_stream("analyst", messages)


def openrouter_think(prompt: str, model: str) -> str | None:
    from memory import JARVIS_PERSONA
    messages = [{"role": "system", "content": JARVIS_PERSONA},
                {"role": "user",   "content": prompt}]
    return _call_model_non_stream("backup", messages)


def ollama_think(prompt: str) -> str | None:
    from memory import JARVIS_PERSONA
    messages = [{"role": "system", "content": JARVIS_PERSONA},
                {"role": "user",   "content": prompt}]
    return _call_model_non_stream("backup", messages)


# ═══════════════════════════════════════════════════════════════════════════════
#  TERMINAL STREAMING (used by main.py)
# ═══════════════════════════════════════════════════════════════════════════════

def stream_response(user_input: str) -> str:
    """Stream AI response to terminal (backward compat for main.py)."""
    result = think(user_input)
    text = result["response"]
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
    print()
    return text