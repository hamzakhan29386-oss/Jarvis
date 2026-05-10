"""
actions.py — JARVIS Agent + Automation Engine
================================================
Full action library + multi-step plan executor + intent parsing.
All tools are 100% free, no paid APIs.

Actions:
    open_app, open_url, web_search, set_timer, take_screenshot,
    get_system_status, type_text, read_clipboard, write_clipboard,
    send_notification, run_script, control_media, get_weather

Usage:
    from actions import execute_action, execute_plan, parse_intent
"""

import os
import sys
import json
import time
import logging
import webbrowser
import subprocess
import threading
import urllib.parse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("jarvis.actions")

# ── Data directory for screenshots etc ──────────────────────────────────────
DATA_DIR = Path(__file__).parent / "data"
SCREENSHOT_DIR = DATA_DIR / "screenshots"


# ═══════════════════════════════════════════════════════════════════════════════
#  ACTION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

ACTION_REGISTRY = {}

def action(name: str, description: str = ""):
    """Decorator to register an action function."""
    def decorator(func):
        ACTION_REGISTRY[name] = {"func": func, "description": description}
        return func
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE ACTIONS (all free)
# ═══════════════════════════════════════════════════════════════════════════════

@action("open_app", "Open an application by name (fuzzy match)")
def open_app(name: str) -> str:
    """Smart app launcher with fuzzy matching on Windows."""
    try:
        name_lower = name.lower().strip()

        # Common app aliases → executable names
        app_aliases = {
            "chrome": "chrome", "google chrome": "chrome",
            "firefox": "firefox", "brave": "brave",
            "edge": "msedge", "microsoft edge": "msedge",
            "notepad": "notepad", "calculator": "calc",
            "paint": "mspaint", "terminal": "wt",
            "cmd": "cmd", "powershell": "powershell",
            "explorer": "explorer", "file explorer": "explorer",
            "vscode": "code", "vs code": "code", "visual studio code": "code",
            "spotify": "spotify", "discord": "discord",
            "notion": "notion", "slack": "slack",
            "task manager": "taskmgr", "settings": "ms-settings:",
            "control panel": "control",
        }

        exe = app_aliases.get(name_lower, name)

        # Try direct launch
        if exe.startswith("ms-"):
            os.startfile(exe)
        else:
            subprocess.Popen(exe, shell=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        log.info(f"[Action] Opened: {name}")
        return f"Opened {name}"

    except Exception as e:
        # Try Start Menu search as fallback
        try:
            subprocess.Popen(f'start "" "{name}"', shell=True)
            return f"Opened {name}"
        except Exception:
            return f"Could not open {name}: {e}"


@action("open_url", "Open a URL in the default browser")
def open_url(url: str, incognito: bool = False) -> str:
    """Open URL, optionally in incognito/private mode."""
    try:
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        if incognito:
            # Try Chrome incognito, Edge InPrivate, Firefox private
            for cmd in [
                f'start chrome --incognito "{url}"',
                f'start msedge --inprivate "{url}"',
                f'start firefox --private-window "{url}"',
            ]:
                try:
                    subprocess.Popen(cmd, shell=True)
                    return f"Opened {url} (private)"
                except Exception:
                    continue

        webbrowser.open(url)
        return f"Opened {url}"
    except Exception as e:
        return f"Could not open URL: {e}"


@action("web_search", "Search DuckDuckGo (free, no API key)")
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo Instant Answer API (free)."""
    try:
        import requests
        # DuckDuckGo Instant Answer API — completely free, no key
        api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1"
        resp = requests.get(api_url, timeout=10)
        data = resp.json()

        results = []

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append(data["AbstractText"])

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(topic["Text"][:200])

        if results:
            return "Search results:\n" + "\n".join(f"• {r}" for r in results)

        # Fallback: open in browser
        search_url = f"https://duckduckgo.com/?q={urllib.parse.quote_plus(query)}"
        webbrowser.open(search_url)
        return f"Opened search for: {query}"

    except Exception as e:
        # Ultimate fallback: Google search in browser
        url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
        webbrowser.open(url)
        return f"Searched: {query}"


@action("set_timer", "Set a countdown timer")
def set_timer(seconds: int, label: str = "Timer") -> str:
    """Set a timer that triggers a notification when done."""
    def _timer_callback():
        try:
            send_notification("JARVIS Timer", f"{label} — Time's up!")
        except Exception:
            log.info(f"[Timer] {label} completed ({seconds}s)")

    timer = threading.Timer(seconds, _timer_callback)
    timer.daemon = True
    timer.start()

    mins = seconds // 60
    secs = seconds % 60
    time_str = f"{mins}m {secs}s" if mins else f"{secs}s"
    return f"Timer set: {label} — {time_str}"


@action("take_screenshot", "Capture the screen")
def take_screenshot() -> str:
    """Take a screenshot and save to data/screenshots/."""
    try:
        from PIL import ImageGrab
        SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = SCREENSHOT_DIR / f"screenshot_{timestamp}.png"
        img = ImageGrab.grab()
        img.save(str(filepath))
        log.info(f"[Action] Screenshot saved: {filepath}")
        return f"Screenshot saved: {filepath.name}"
    except ImportError:
        return "Pillow not installed. Run: pip install Pillow"
    except Exception as e:
        return f"Screenshot failed: {e}"


@action("get_system_status", "Get CPU, RAM, disk, and battery info")
def get_system_status() -> dict:
    """Return system resource usage via psutil."""
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.5)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        status = {
            "cpu_percent": cpu_percent,
            "ram_total_gb": round(mem.total / (1024**3), 1),
            "ram_used_gb": round(mem.used / (1024**3), 1),
            "ram_percent": mem.percent,
            "disk_total_gb": round(disk.total / (1024**3), 1),
            "disk_used_gb": round(disk.used / (1024**3), 1),
            "disk_percent": round(disk.percent, 1),
        }

        # Battery (if laptop)
        battery = psutil.sensors_battery()
        if battery:
            status["battery_percent"] = battery.percent
            status["battery_plugged"] = battery.power_plugged
        else:
            status["battery_percent"] = None

        return status

    except ImportError:
        return {"error": "psutil not installed. Run: pip install psutil"}
    except Exception as e:
        return {"error": str(e)}


@action("type_text", "Type text at the current cursor position")
def type_text(text: str) -> str:
    """Simulate typing at the cursor using pyautogui."""
    try:
        import pyautogui
        pyautogui.typewrite(text, interval=0.02)
        return f"Typed: {text[:50]}..."
    except ImportError:
        return "pyautogui not installed. Run: pip install pyautogui"
    except Exception as e:
        return f"Typing failed: {e}"


@action("read_clipboard", "Read text from the clipboard")
def read_clipboard() -> str:
    """Read the current clipboard content."""
    try:
        import pyperclip
        content = pyperclip.paste()
        return content if content else "(clipboard is empty)"
    except ImportError:
        return "pyperclip not installed. Run: pip install pyperclip"
    except Exception as e:
        return f"Clipboard read failed: {e}"


@action("write_clipboard", "Write text to the clipboard")
def write_clipboard(text: str) -> str:
    """Copy text to the clipboard."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return f"Copied to clipboard: {text[:50]}..."
    except ImportError:
        return "pyperclip not installed. Run: pip install pyperclip"
    except Exception as e:
        return f"Clipboard write failed: {e}"


@action("send_notification", "Show a desktop notification")
def send_notification(title: str, body: str) -> str:
    """Send a desktop notification using plyer (cross-platform, free)."""
    try:
        from plyer import notification
        notification.notify(
            title=title, message=body, timeout=5, app_name="JARVIS"
        )
        return f"Notification sent: {title}"
    except ImportError:
        # Fallback: Windows toast via PowerShell
        try:
            ps_cmd = (
                f'powershell -Command "'
                f"[Windows.UI.Notifications.ToastNotificationManager, "
                f"Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null; "
                f"$template = [Windows.UI.Notifications.ToastNotificationManager]::"
                f"GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::"
                f"ToastText02); $template.GetElementsByTagName('text')[0].AppendChild("
                f"$template.CreateTextNode('{title}')); "
                f"$template.GetElementsByTagName('text')[1].AppendChild("
                f"$template.CreateTextNode('{body}')); "
                f"[Windows.UI.Notifications.ToastNotificationManager]::"
                f'CreateToastNotifier(\'JARVIS\').Show($template)"'
            )
            subprocess.Popen(ps_cmd, shell=True)
            return f"Notification sent: {title}"
        except Exception:
            log.info(f"[Notification] {title}: {body}")
            return f"Notification (logged): {title}"
    except Exception as e:
        return f"Notification failed: {e}"


@action("run_script", "Execute a Python script")
def run_script(path: str) -> str:
    """Run a user-defined Python script in a subprocess."""
    try:
        script_path = Path(path)
        if not script_path.exists():
            return f"Script not found: {path}"
        if not str(script_path).endswith(".py"):
            return "Only .py scripts are allowed for security."

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout.strip() if result.stdout else ""
        errors = result.stderr.strip() if result.stderr else ""

        if result.returncode == 0:
            return f"Script completed. Output: {output[:300]}"
        else:
            return f"Script error: {errors[:300]}"

    except subprocess.TimeoutExpired:
        return "Script timed out (30s limit)."
    except Exception as e:
        return f"Script execution failed: {e}"


@action("control_media", "Play/pause/next/prev/volume media controls")
def control_media(action_name: str) -> str:
    """Control media playback via keyboard simulation."""
    try:
        import pyautogui

        media_keys = {
            "play": "playpause", "pause": "playpause",
            "play_pause": "playpause", "playpause": "playpause",
            "next": "nexttrack", "skip": "nexttrack",
            "prev": "prevtrack", "previous": "prevtrack",
            "volume_up": "volumeup", "louder": "volumeup",
            "volume_down": "volumedown", "quieter": "volumedown",
            "mute": "volumemute",
        }

        key = media_keys.get(action_name.lower().strip())
        if key:
            pyautogui.press(key)
            return f"Media: {action_name}"
        return f"Unknown media action: {action_name}"

    except ImportError:
        return "pyautogui not installed."
    except Exception as e:
        return f"Media control failed: {e}"


@action("get_weather", "Get weather via wttr.in (free, no API key)")
def get_weather(city: str = "") -> str:
    """Fetch weather from wttr.in (completely free, no key needed)."""
    try:
        import requests
        url = f"https://wttr.in/{urllib.parse.quote(city)}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        current = data["current_condition"][0]
        desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        temp_c = current.get("temp_C", "?")
        humidity = current.get("humidity", "?")
        wind = current.get("windspeedKmph", "?")
        return (
            f"Weather in {city or 'your location'}: {desc}, "
            f"{temp_c}°C, humidity {humidity}%, wind {wind} km/h"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  BACKWARD-COMPATIBLE EXECUTE (from original actions.py)
# ═══════════════════════════════════════════════════════════════════════════════

def execute(command: str) -> str:
    """
    Execute an action based on a command string.
    Backward-compatible with the original actions.py interface.
    """
    if not command:
        return ""
    cmd_lower = command.strip().lower()

    if cmd_lower.startswith("open "):
        target = cmd_lower[5:].strip()
        if "." in target or target.startswith("http"):
            return open_url(target)
        return open_app(target)

    if cmd_lower.startswith("search "):
        return web_search(command[7:].strip())

    if cmd_lower.startswith("timer "):
        parts = command[6:].strip().split(" ", 1)
        try:
            secs = int(parts[0])
            label = parts[1] if len(parts) > 1 else "Timer"
            return set_timer(secs, label)
        except ValueError:
            return f"Invalid timer duration: {parts[0]}"

    if cmd_lower == "screenshot":
        return take_screenshot()

    if cmd_lower == "status" or cmd_lower == "system status":
        status = get_system_status()
        return json.dumps(status, indent=2) if isinstance(status, dict) else str(status)

    return f"Unknown command: {command}"


# ═══════════════════════════════════════════════════════════════════════════════
#  EXECUTE ACTION (by name)
# ═══════════════════════════════════════════════════════════════════════════════

def execute_action(action_name: str, args: dict = None) -> str:
    """
    Execute a registered action by name with arguments.

    Args:
        action_name: Name of the action (e.g., "open_app")
        args: Dictionary of keyword arguments.

    Returns:
        Result string from the action.
    """
    if args is None:
        args = {}

    entry = ACTION_REGISTRY.get(action_name)
    if not entry:
        return f"Unknown action: {action_name}"

    try:
        result = entry["func"](**args)
        log.info(f"[Action] {action_name} → {str(result)[:80]}")
        return result if isinstance(result, str) else json.dumps(result)
    except Exception as e:
        log.error(f"[Action] {action_name} failed: {e}")
        return f"Action '{action_name}' failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  AGENT PLAN EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanStep:
    """A single step in an agent plan."""
    action: str
    args: dict = field(default_factory=dict)
    delay_ms: int = 0
    status: str = "pending"  # pending | running | success | failed | skipped
    result: str = ""
    retries: int = 1


@dataclass
class AgentPlan:
    """A multi-step execution plan."""
    plan_id: str
    label: str
    steps: list = field(default_factory=list)
    on_complete: str = ""
    status: str = "pending"  # pending | running | completed | partial


def execute_plan(plan: AgentPlan, progress_callback=None) -> dict:
    """
    Execute a multi-step agent plan with self-correction.

    Features:
        - Sequential execution with delays
        - Retry on failure (once per step)
        - Skip failed steps and continue
        - Final summary report

    Args:
        plan: The AgentPlan to execute.
        progress_callback: Optional callable(step_index, step) for UI updates.

    Returns:
        dict with execution summary.
    """
    plan.status = "running"
    total = len(plan.steps)
    completed = 0
    failed_steps = []

    log.info(f"[Agent] Executing plan: {plan.label} ({total} steps)")

    for i, step in enumerate(plan.steps):
        step.status = "running"
        if progress_callback:
            progress_callback(i, step)

        # Apply delay
        if step.delay_ms > 0:
            time.sleep(step.delay_ms / 1000.0)

        # Execute with retry
        success = False
        for attempt in range(1 + step.retries):
            try:
                result = execute_action(step.action, step.args)
                if "failed" not in result.lower() and "error" not in result.lower():
                    step.status = "success"
                    step.result = result
                    completed += 1
                    success = True
                    break
                else:
                    if attempt < step.retries:
                        log.warning(f"[Agent] Step {i+1} failed, retrying...")
                        time.sleep(0.5)
                    else:
                        step.status = "failed"
                        step.result = result
            except Exception as e:
                if attempt < step.retries:
                    log.warning(f"[Agent] Step {i+1} error: {e}, retrying...")
                    time.sleep(0.5)
                else:
                    step.status = "failed"
                    step.result = str(e)

        if not success:
            step.status = "skipped" if step.status != "failed" else "failed"
            failed_steps.append(f"{step.action} ({step.result})")

        if progress_callback:
            progress_callback(i, step)

    # Final status
    plan.status = "completed" if completed == total else "partial"

    summary = {
        "plan_id": plan.plan_id,
        "label": plan.label,
        "total_steps": total,
        "completed": completed,
        "failed": total - completed,
        "status": plan.status,
        "failed_details": failed_steps,
        "on_complete": plan.on_complete,
        "steps": [
            {"action": s.action, "status": s.status, "result": s.result}
            for s in plan.steps
        ],
    }

    # Build JARVIS-style report
    if completed == total:
        summary["report"] = f"All {total} steps completed successfully, sir."
    else:
        summary["report"] = (
            f"{completed} of {total} steps completed. "
            f"Failed: {', '.join(failed_steps)}"
        )

    log.info(f"[Agent] Plan complete: {summary['report']}")
    return summary


def build_plan_from_procedure(procedure: dict) -> AgentPlan:
    """
    Convert a stored procedure (from memory.py) into an executable AgentPlan.

    Args:
        procedure: Dict with 'name' and 'steps' (list of action strings).

    Returns:
        AgentPlan ready for execution.
    """
    steps = []
    for i, step_str in enumerate(procedure.get("steps", [])):
        step_lower = step_str.lower().strip()

        # Parse the step string into action + args
        if step_lower.startswith("open "):
            target = step_str[5:].strip()
            if "." in target or "http" in target:
                steps.append(PlanStep(action="open_url", args={"url": target}, delay_ms=i*1000))
            else:
                steps.append(PlanStep(action="open_app", args={"name": target}, delay_ms=i*1000))
        elif step_lower.startswith("search "):
            steps.append(PlanStep(action="web_search", args={"query": step_str[7:].strip()}, delay_ms=500))
        elif step_lower.startswith("set timer") or step_lower.startswith("timer"):
            # Try to extract duration
            import re
            nums = re.findall(r'\d+', step_str)
            duration = int(nums[0]) if nums else 1800
            steps.append(PlanStep(action="set_timer", args={"seconds": duration, "label": step_str}, delay_ms=500))
        elif step_lower.startswith("notify") or step_lower.startswith("notification"):
            msg = step_str.split(" ", 1)[1] if " " in step_str else step_str
            steps.append(PlanStep(action="send_notification", args={"title": "JARVIS", "body": msg}, delay_ms=500))
        else:
            # Generic: try as app name
            steps.append(PlanStep(action="open_app", args={"name": step_str}, delay_ms=i*800))

    return AgentPlan(
        plan_id=procedure.get("name", "unnamed"),
        label=procedure.get("name", "Custom Plan").replace("_", " ").title(),
        steps=steps,
        on_complete=f"Procedure '{procedure.get('name', 'custom')}' executed.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  INTENT PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def parse_intent(ai_response: str) -> dict:
    """
    Parse an AI response for actionable intents.
    Uses keyword detection (free, no API call).

    Returns:
        dict: {
            "type": "single_action" | "multi_step_plan" | "pure_response",
            "actions": [{"action": str, "args": dict}],
            "requires_confirmation": bool,
            "confidence": float
        }
    """
    resp_lower = ai_response.lower()
    actions = []
    intent_type = "pure_response"
    confidence = 0.3

    # Detect action patterns in response
    action_patterns = [
        (r"(?:let me |i'll |opening |launching )open (\S+)", "open_app"),
        (r"(?:searching|let me search|looking up) (?:for )?(.+?)(?:\.|$)", "web_search"),
        (r"(?:setting|set) (?:a )?timer (?:for )?(\d+)", "set_timer"),
        (r"(?:taking|captured) (?:a )?screenshot", "take_screenshot"),
        (r"(?:checking|getting) system (?:status|info)", "get_system_status"),
    ]

    import re
    for pattern, action_name in action_patterns:
        match = re.search(pattern, resp_lower)
        if match:
            args = {}
            if action_name == "open_app":
                args = {"name": match.group(1)}
            elif action_name == "web_search":
                args = {"query": match.group(1).strip()}
            elif action_name == "set_timer":
                args = {"seconds": int(match.group(1)), "label": "Timer"}
            actions.append({"action": action_name, "args": args})

    if len(actions) > 1:
        intent_type = "multi_step_plan"
        confidence = 0.7
    elif len(actions) == 1:
        intent_type = "single_action"
        confidence = 0.8

    return {
        "type": intent_type,
        "actions": actions,
        "requires_confirmation": len(actions) > 2,
        "confidence": confidence,
    }


def get_available_actions() -> list:
    """Return a list of all registered actions with descriptions."""
    return [
        {"name": name, "description": info["description"]}
        for name, info in ACTION_REGISTRY.items()
    ]
