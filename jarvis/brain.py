"""
brain.py — JARVIS Cognitive Architecture (Hybrid Local + Cloud)
================================================================
HYBRID ARCHITECTURE:
  - reflex  → Ollama local (llama3.2:3b) — instant, no latency
  - analyst → NVIDIA NIM cloud
  - coder   → OpenRouter cloud
  - oracle  → NVIDIA NIM cloud
  - ultra   → NVIDIA NIM cloud
  - backup  → OpenRouter cloud fallback

Ollama powers fast conversational interactions locally.
Cloud models handle intelligence-heavy tasks.

Usage:
    from brain import think, think_stream, set_mode, get_mode
"""

import os
import sys
import json
import logging
import time
import threading

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("jarvis.brain")

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NVIDIA_API_KEY   = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_KEY   = os.getenv("OPENROUTER_API_KEY", "")

NVIDIA_BASE_URL     = "https://integrate.api.nvidia.com/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_BASE_URL     = "http://localhost:11434/v1"   # ← local Ollama

MODELS = {

    # ── LOCAL (Ollama) — zero cloud latency ─────────────────────────────
    "reflex": {
        "provider":    "ollama",
        "model":       "llama3.2:3b",
        "use_for":     "greetings, commands, quick answers, voice replies",
        "max_tokens":  256,
        "temperature": 0.7,
    },

    # ── CLOUD — intelligence tasks ───────────────────────────────────────
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
        "use_for":     "cloud fallback when all else fails",
        "max_tokens":  1024,
        "temperature": 0.7,
    },
    "backup2": {
        "provider":    "openrouter",
        "model":       "google/gemma-3-4b-it:free",
        "use_for":     "second cloud fallback",
        "max_tokens":  512,
        "temperature": 0.7,
    },
    "backup3": {
        "provider":    "openrouter",
        "model":       "mistralai/mistral-small-3.1:free",
        "use_for":     "third cloud fallback",
        "max_tokens":  512,
        "temperature": 0.7,
    },
}

# Reflex falls back to cloud backup if Ollama is offline
FALLBACK_CHAINS = {
    "reflex":  ["reflex",  "backup",  "backup2", "backup3"],
    "analyst": ["analyst", "reflex",  "backup",  "backup2", "backup3"],
    "coder":   ["coder",   "oracle",  "analyst", "backup",  "backup2", "backup3"],
    "oracle":  ["oracle",  "analyst", "backup",  "backup2", "backup3"],
    "ultra":   ["ultra",   "oracle",  "analyst", "backup",  "backup2", "backup3"],
    "backup":  ["backup",  "backup2", "backup3"],
}

CURRENT_MODE = "auto"

_is_generating = False
_gen_lock = threading.Lock()

_session_stats = {
    "reflex": 0, "analyst": 0, "coder": 0, "oracle": 0,
    "ultra": 0, "backup": 0, "total": 0, "fallbacks": 0,
    "local_calls": 0, "cloud_calls": 0,
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
#  QUERY CLASSIFICATION (unchanged)
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
    """Zero-cost keyword classifier."""
    q = query.lower().strip()

    if len(q) < 20 or any(kw in q for kw in _REFLEX_KEYWORDS):
        if not any(kw in q for kw in _CODER_KEYWORDS + _ANALYST_KEYWORDS + _ORACLE_KEYWORDS):
            log.info("[JARVIS CORE] → Classified: reflex (local)")
            return "reflex"

    for p in _ULTRA_KEYWORDS:
        if p in q:
            log.info("[JARVIS CORE] → Classified: ultra (cloud)")
            return "ultra"

    if any(kw in q for kw in _CODER_KEYWORDS):
        log.info("[JARVIS CORE] → Classified: coder (cloud)")
        return "coder"

    for p in _ORACLE_KEYWORDS:
        if p in q:
            log.info("[JARVIS CORE] → Classified: oracle (cloud)")
            return "oracle"

    if any(kw in q for kw in _ANALYST_KEYWORDS) or len(query) > 100:
        log.info("[JARVIS CORE] → Classified: analyst (cloud)")
        return "analyst"

    log.info("[JARVIS CORE] → Classified: analyst (cloud, default)")
    return "analyst"


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CLIENT FACTORY
# ═══════════════════════════════════════════════════════════════════════════════

def _make_client(provider: str) -> OpenAI | None:
    """Build an OpenAI-compatible client for the given provider."""

    if provider == "ollama":
        # Local Ollama — always available, no key required
        return OpenAI(
            api_key="ollama",          # Ollama ignores the key
            base_url=OLLAMA_BASE_URL,
            max_retries=0,
            timeout=8.0,               # Local is fast; 8s is generous
        )

    elif provider == "nvidia":
        if not NVIDIA_API_KEY:
            log.warning("[JARVIS] NVIDIA key missing — will cascade")
            return None
        return OpenAI(
            api_key=NVIDIA_API_KEY,
            base_url=NVIDIA_BASE_URL,
            max_retries=0,
            timeout=15.0,
        )

    else:  # openrouter
        if not OPENROUTER_KEY:
            log.warning("[JARVIS] OpenRouter key missing — will cascade")
            return None
        return OpenAI(
            api_key=OPENROUTER_KEY,
            base_url=OPENROUTER_BASE_URL,
            max_retries=0,
            timeout=20.0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL CALLERS
# ═══════════════════════════════════════════════════════════════════════════════

def _call_model_non_stream(tier: str, messages: list) -> str | None:
    cfg = MODELS.get(tier, MODELS["backup"])
    client = _make_client(cfg["provider"])

    if client is None:
        if tier != "backup":
            return _call_model_non_stream("backup", messages)
        return "API keys not configured, sir."

    log.info(
        f"[JARVIS CORE] → Tier: {tier} | Model: {cfg['model']} "
        f"| Provider: {cfg['provider']} ({'LOCAL' if cfg['provider']=='ollama' else 'CLOUD'})"
    )

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg.get("temperature", 0.7),
            stream=False,
        )
        return response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            log.warning(f"[JARVIS] {tier} rate limited — cascading immediately")
        elif "Connection refused" in err or "ConnectError" in err:
            log.warning(f"[JARVIS] {tier} unreachable (Ollama running?) — cascading")
        else:
            log.error(f"[JARVIS] {tier} failed: {err}")
        return None


def _call_model_stream(tier: str, messages: list):
    """Streaming model caller. Yields token strings."""
    cfg = MODELS.get(tier, MODELS["backup"])
    client = _make_client(cfg["provider"])

    if client is None:
        if tier != "backup":
            yield from _call_model_stream("backup", messages)
        else:
            yield "API keys not configured, sir."
        return

    log.info(
        f"[JARVIS CORE] → Streaming: {tier} | {cfg['model']} "
        f"| {'LOCAL' if cfg['provider']=='ollama' else 'CLOUD'}"
    )

    try:
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=messages,
            max_tokens=cfg["max_tokens"],
            temperature=cfg.get("temperature", 0.7),
            stream=True,
        )
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        err = str(e)
        if "429" in err or "Too Many Requests" in err:
            log.warning(f"[JARVIS] {tier} rate limited — cascading")
        elif "Connection refused" in err or "ConnectError" in err:
            log.warning(f"[JARVIS] {tier} (Ollama) unreachable — cascading to cloud")
        else:
            log.error(f"[JARVIS] {tier} stream failed: {err}")

        if tier != "backup":
            log.info(f"[JARVIS] Cascading from {tier} to next in chain...")
            yield from _call_model_stream("backup", messages)
        else:
            yield "All providers currently unavailable, sir."


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
        "local_model": MODELS["reflex"]["model"],
        "ollama_url": OLLAMA_BASE_URL,
    }


def _resolve_model_name() -> str:
    if CURRENT_MODE == "fast":
        return MODELS["reflex"]["model"] + " (local)"
    elif CURRENT_MODE == "smart":
        return MODELS["oracle"]["model"]
    return "auto (local+cloud)"


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
        provider = MODELS.get(tier, {}).get("provider", "")
        if provider == "ollama":
            _session_stats["local_calls"] += 1
        else:
            _session_stats["cloud_calls"] += 1


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN THINK (non-streaming)
# ═══════════════════════════════════════════════════════════════════════════════

def think(prompt: str) -> dict:
    """Full JARVIS cognitive pipeline (non-streaming)."""
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
                f"{', '.join(remember_cmd['steps'])}."
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

        response = _call_model_non_stream(model_key, messages)
        if response:
            used_model = model_info["model"]
            used_tier = model_key
            _record_stat(model_key, is_fallback=fallback_used)
            break

    if not response:
        response = (
            "All AI engines are currently offline, sir. "
            "Ensure Ollama is running and API keys are set in .env."
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
    provider = MODELS.get(used_tier, {}).get("provider", "unknown")
    log.info(
        f"[PERF] think() | Model: {used_model} | Tier: {used_tier} "
        f"| Provider: {provider} | Time: {t_elapsed:.2f}s | Fallback: {fallback_used}"
    )

    return {
        "action": "NONE",
        "response": response,
        "model": used_model,
        "fallback": fallback_used,
        "tier": used_tier,
        "provider": provider,
        "memory_used": memory_used,
        "response_time_ms": int(t_elapsed * 1000),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMING THINK
# ═══════════════════════════════════════════════════════════════════════════════

def think_stream(prompt: str):
    """
    Generator — yields tokens as they arrive for SSE streaming.
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
                            f"[PERF] First token: "
                            f"{(first_token_time - t_start)*1000:.0f}ms "
                            f"({'LOCAL' if model_info['provider']=='ollama' else 'CLOUD'})"
                        )
                    full_response += token
                    yield {"token": token}
                    streamed = True
            finally:
                with _gen_lock:
                    _is_generating = False

            if streamed:
                t_elapsed = time.time() - t_start
                provider = model_info["provider"]
                log.info(
                    f"[PERF] stream() | {model_info['model']} "
                    f"| {provider.upper()} | {t_elapsed:.2f}s | {len(full_response)} chars"
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
                    "provider": provider,
                    "fallback": fallback_used,
                    "memory_used": memory_used,
                    "response_time_ms": int(t_elapsed * 1000),
                }
                return

        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many Requests" in err:
                log.warning(f"[JARVIS] {model_key} rate limited — trying next")
            elif "Connection refused" in err or "ConnectError" in err:
                log.warning(f"[JARVIS] {model_key} unreachable — trying next")
            else:
                log.error(f"[JARVIS] {model_key} stream exception: {err}")
            continue

    error_msg = "All engines offline, sir. Check Ollama and API keys."
    for ch in error_msg:
        yield {"token": ch}
    yield {"done": True, "model": "none", "tier": "none",
           "provider": "none", "fallback": True, "memory_used": False}


# ═══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> bool:
    """Check if local Ollama is running and llama3.2:3b is available."""
    try:
        import requests
        r = requests.get(
            f"{OLLAMA_BASE_URL.replace('/v1', '')}/api/tags",
            timeout=2.0
        )
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            available = any("llama3.2:3b" in m or "llama3.2" in m for m in models)
            if not available:
                log.info("[Ollama] Running but llama3.2:3b not pulled. Run: ollama pull llama3.2:3b")
            return True
        return False
    except Exception:
        return False


def check_ollama_local() -> bool:
    """Alias for backward compat."""
    return check_ollama()


def check_nvidia() -> bool:
    return bool(NVIDIA_API_KEY)


def check_openrouter() -> bool:
    return bool(OPENROUTER_KEY)


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
    """Now actually calls local Ollama reflex tier."""
    from memory import JARVIS_PERSONA
    messages = [{"role": "system", "content": JARVIS_PERSONA},
                {"role": "user",   "content": prompt}]
    return _call_model_non_stream("reflex", messages)


def stream_response(user_input: str) -> str:
    """Stream AI response to terminal (backward compat for main.py)."""
    result = think(user_input)
    text = result["response"]
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
    print()
    return text