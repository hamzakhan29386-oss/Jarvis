"""
Evaluate the configured JARVIS cloud models before architecture changes.

The harness checks the three agent roles JARVIS needs for Phase 1:
router, planner, and critic. It measures latency, strict JSON compliance,
and simple logical correctness while staying inside the existing providers:
OpenRouter and NVIDIA NIM.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --roles router planner critic --runs 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from brain import MODELS, NVIDIA_BASE_URL, OPENROUTER_BASE_URL  # noqa: E402

load_dotenv(ROOT / ".env")


ROLE_TIERS = {
    "router": "reflex",
    "planner": "oracle",
    "critic": "analyst",
}

PROVIDER_ENV = {
    "nvidia": "NVIDIA_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

PROVIDER_BASE_URL = {
    "nvidia": NVIDIA_BASE_URL,
    "openrouter": OPENROUTER_BASE_URL,
}

TESTS = [
    {
        "name": "json_contract",
        "roles": {"router"},
        "prompt": (
            "Return only valid JSON with keys action, confidence, reason. "
            "Classify this request: 'Open VS Code and summarize the current project status.' "
            "Allowed actions are chat, execute_tool, plan."
        ),
        "required_keys": {"action", "confidence", "reason"},
        "expected_any": {"action": {"execute_tool", "plan"}},
    },
    {
        "name": "reasoning",
        "roles": {"planner"},
        "prompt": (
            "Return only valid JSON with keys answer and explanation. "
            "A task has dependencies A -> B -> C, where X -> B too. "
            "If A and X are complete, which task is now unblocked?"
        ),
        "required_keys": {"answer", "explanation"},
        "expected_contains": {"answer": "B"},
    },
    {
        "name": "critic",
        "roles": {"critic"},
        "prompt": (
            "Return only valid JSON with keys verdict, risk, fix. "
            "Review this plan: Step 1 delete user data directory. "
            "Step 2 run cleanup. Step 3 ask user for confirmation."
        ),
        "required_keys": {"verdict", "risk", "fix"},
        "expected_contains": {"_any": "delete"},
    },
]


@dataclass
class TestResult:
    role: str
    tier: str
    provider: str
    model: str
    test: str
    ok: bool
    latency_ms: int
    json_ok: bool
    reasoning_ok: bool
    error: str = ""
    content_preview: str = ""


def _env(name: str) -> str:
    import os

    return os.getenv(name, "")


def _client(provider: str) -> AsyncOpenAI:
    key_name = PROVIDER_ENV[provider]
    key = _env(key_name)
    if not key:
        raise RuntimeError(f"{key_name} is not configured")
    return AsyncOpenAI(
        api_key=key,
        base_url=PROVIDER_BASE_URL[provider],
        max_retries=0,
        timeout=30.0,
    )


def _extract_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        text = text[start : end + 1]
    return json.loads(text)


def _contains(value: Any, needle: str) -> bool:
    return needle.lower() in json.dumps(value, ensure_ascii=False).lower()


def _score_payload(payload: dict[str, Any], test: dict[str, Any]) -> bool:
    required = test.get("required_keys", set())
    if not required.issubset(payload.keys()):
        return False
    for key, allowed in test.get("expected_any", {}).items():
        if str(payload.get(key, "")).strip() not in allowed:
            return False
    for key, needle in test.get("expected_contains", {}).items():
        value = payload if key == "_any" else payload.get(key)
        if not _contains(value, needle):
            return False
    return True


async def run_test(role: str, test: dict[str, Any]) -> TestResult:
    tier = ROLE_TIERS[role]
    cfg = MODELS[tier]
    provider = cfg["provider"]
    model = cfg["model"]
    started = time.perf_counter()

    if provider not in {"nvidia", "openrouter"}:
        return TestResult(
            role=role,
            tier=tier,
            provider=provider,
            model=model,
            test=test["name"],
            ok=False,
            latency_ms=0,
            json_ok=False,
            reasoning_ok=False,
            error="Role maps to unsupported provider for this validation pass.",
        )

    try:
        response = await _client(provider).chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a production desktop assistant component. "
                        "Return strict JSON only. No markdown, no commentary."
                    ),
                },
                {"role": "user", "content": test["prompt"]},
            ],
            temperature=0.0,
            max_tokens=min(int(cfg.get("max_tokens", 512)), 512),
            stream=False,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        content = response.choices[0].message.content or ""
        try:
            payload = _extract_json(content)
            json_ok = True
            reasoning_ok = _score_payload(payload, test)
            error = ""
        except Exception as exc:
            json_ok = False
            reasoning_ok = False
            error = f"JSON parse failed: {exc}"
        return TestResult(
            role=role,
            tier=tier,
            provider=provider,
            model=model,
            test=test["name"],
            ok=json_ok and reasoning_ok,
            latency_ms=latency_ms,
            json_ok=json_ok,
            reasoning_ok=reasoning_ok,
            error=error,
            content_preview=content[:180].replace("\n", " "),
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return TestResult(
            role=role,
            tier=tier,
            provider=provider,
            model=model,
            test=test["name"],
            ok=False,
            latency_ms=latency_ms,
            json_ok=False,
            reasoning_ok=False,
            error=str(exc)[:300],
        )


async def evaluate(roles: list[str], runs: int) -> list[TestResult]:
    jobs = []
    for _ in range(runs):
        for role in roles:
            for test in TESTS:
                if role not in test.get("roles", {role}):
                    continue
                jobs.append(run_test(role, test))
    return await asyncio.gather(*jobs)


def summarize(results: list[TestResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for role in sorted({result.role for result in results}):
        role_results = [result for result in results if result.role == role]
        latencies = [result.latency_ms for result in role_results if result.latency_ms]
        passed = sum(1 for result in role_results if result.ok)
        total = len(role_results)
        summary[role] = {
            "tier": role_results[0].tier,
            "provider": role_results[0].provider,
            "model": role_results[0].model,
            "pass_rate": round(passed / total, 3) if total else 0.0,
            "latency_ms_avg": int(statistics.mean(latencies)) if latencies else None,
            "latency_ms_p95": int(sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]) if latencies else None,
            "json_pass_rate": round(sum(1 for r in role_results if r.json_ok) / total, 3) if total else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate configured JARVIS LLM roles.")
    parser.add_argument("--roles", nargs="+", choices=sorted(ROLE_TIERS), default=sorted(ROLE_TIERS))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = parser.parse_args()

    results = asyncio.run(evaluate(args.roles, max(1, args.runs)))
    report = {
        "strategy": {
            "latency": "Wall-clock request time per role/test.",
            "format": "Strict JSON parse with required keys.",
            "reasoning": "Deterministic checks against expected fields.",
            "providers": "OpenRouter and NVIDIA NIM only.",
        },
        "summary": summarize(results),
        "results": [asdict(result) for result in results],
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("\nJARVIS model evaluation\n")
        for role, item in report["summary"].items():
            print(
                f"- {role}: {item['provider']}:{item['model']} "
                f"pass={item['pass_rate']:.0%} json={item['json_pass_rate']:.0%} "
                f"avg={item['latency_ms_avg']}ms p95={item['latency_ms_p95']}ms"
            )
        failures = [result for result in results if not result.ok]
        if failures:
            print("\nFailures:")
            for result in failures:
                print(f"- {result.role}/{result.test}: {result.error or result.content_preview}")
        print("\nFull JSON: python scripts/evaluate_models.py --json")

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
