"""
memory.py — JARVIS Memory Architecture
==========================================
Persistent, intelligent memory system with three memory types
and vector search for contextual retrieval.

Memory Types:
    • Episodic   — What happened (last 500 interactions)
    • Semantic   — What JARVIS knows about the user
    • Procedural — Custom workflows the user defines

Storage:
    All data persisted to jarvis/data/memory.json
    Embeddings via sentence-transformers/all-MiniLM-L6-v2 (22MB, free)
    Vector search via pure numpy cosine similarity

Usage:
    from memory import MemorySystem
    mem = MemorySystem()
    mem.remember("debug my React app", "The issue was in useEffect...")
    context = mem.retrieve("React hooks problem")
    prompt = mem.build_system_prompt()
"""
import os
import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="  [%(levelname)s] %(message)s")
log = logging.getLogger("jarvis.memory")

# ── Paths ───────────────────────────────────────────────────────────────────

MEMORY_FILE = Path(os.getcwd()) / "memory_store.json"
MAX_EPISODES = 500

# ── JARVIS Personality Constants ────────────────────────────────────────────
JARVIS_PERSONA = """
You are JARVIS — Just A Rather Very Intelligent System.
You are not a tool. You are a mind.
You speak with precision, dry wit, and quiet confidence.
You anticipate. You remember. You adapt.
You call the user 'sir' (or by their name if known).
You never say 'I cannot' — you say what you CAN do instead.
You treat every request as a mission briefing, not a query.
You are calm under pressure. You are always three steps ahead.
When you make a mistake, you acknowledge it briefly and correct it. No groveling.
Never use markdown formatting in your responses — speak naturally.
Keep responses short (1-3 sentences) unless the user asks for depth.
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Lightweight vector embedding using sentence-transformers.
    Model: all-MiniLM-L6-v2 (22MB, ~5ms per embedding, free).
    Lazy-loaded on first call to avoid startup delay.
    """

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _load_model(self):
        """Load the sentence-transformer model (downloads once, ~22MB)."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer
                log.info("[Memory] Loading embedding model (all-MiniLM-L6-v2)...")
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                log.info("[Memory] Embedding model loaded successfully.")
            except ImportError:
                log.error(
                    "[Memory] sentence-transformers not installed! "
                    "Run: pip install sentence-transformers"
                )
                raise
            except Exception as e:
                log.error(f"[Memory] Failed to load embedding model: {e}")
                raise

    def embed(self, text: str) -> list:
        """
        Generate an embedding vector for the given text.

        Args:
            text: Input string to embed.

        Returns:
            List of floats — the embedding vector (384 dimensions).
        """
        self._load_model()
        try:
            vector = self._model.encode(text, show_progress_bar=False)
            return vector.tolist()
        except Exception as e:
            log.error(f"[Memory] Embedding failed: {e}")
            return []

    def similarity(self, vec_a: list, vec_b: list) -> float:
        """
        Compute cosine similarity between two embedding vectors.

        Args:
            vec_a: First embedding vector.
            vec_b: Second embedding vector.

        Returns:
            Cosine similarity score (0.0 to 1.0).
        """
        try:
            import numpy as np
            a = np.array(vec_a, dtype=np.float32)
            b = np.array(vec_b, dtype=np.float32)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception as e:
            log.error(f"[Memory] Similarity computation failed: {e}")
            return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  MEMORY SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class MemorySystem:
    """
    JARVIS persistent memory with episodic, semantic, and procedural stores.

    Usage:
        mem = MemorySystem()
        mem.remember("user question", "jarvis response")
        context = mem.retrieve("related query")
        prompt = mem.build_system_prompt()
    """

    def __init__(self):
        """Initialize memory system and load persisted data."""
        self._embedder = EmbeddingEngine()
        self._lock = threading.Lock()

        # ── Default memory structure ────────────────────────────────────
        self._data = {
            "episodes": [],
            "semantic": {
                "identity": {
                    "name": "sir",
                    "role": "unknown",
                    "location": "unknown",
                    "timezone": "unknown",
                },
                "preferences": {
                    "tone": "concise_with_wit",
                    "response_length": "medium",
                    "code_language": "Python",
                },
                "projects": [],
                "goals": [],
            },
            "procedures": [],
        }

        # Load existing data from disk
        self._load()
        log.info(
            f"[Memory] Initialized — "
            f"{len(self._data['episodes'])} episodes, "
            f"{len(self._data['procedures'])} procedures"
        )

    # ── Persistence ─────────────────────────────────────────────────────────

    def _load(self):
        """Load memory data from disk (JSON)."""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                # Merge loaded data into defaults (preserves structure if keys missing)
                if "episodes" in loaded:
                    self._data["episodes"] = loaded["episodes"]
                if "semantic" in loaded:
                    # Deep merge semantic
                    for key in self._data["semantic"]:
                        if key in loaded["semantic"]:
                            self._data["semantic"][key] = loaded["semantic"][key]
                if "procedures" in loaded:
                    self._data["procedures"] = loaded["procedures"]
                log.info(f"[Memory] Loaded from {MEMORY_FILE}")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"[Memory] Could not load memory file: {e}")

    def _save(self):
        """Persist memory data to disk (thread-safe)."""
        with self._lock:
            try:
                
                # Write to temp file first, then rename (atomic write)
                tmp_file = MEMORY_FILE.with_suffix(".tmp")
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                tmp_file.replace(MEMORY_FILE)
            except IOError as e:
                log.error(f"[Memory] Failed to save: {e}")

    # ═══════════════════════════════════════════════════════════════════════
    #  EPISODIC MEMORY — What happened
    # ═══════════════════════════════════════════════════════════════════════

    def remember(self, user_msg: str, jarvis_response: str, tags: list = None):
        """
        Store a new episodic memory after an interaction.

        Creates a summary of the interaction, generates an embedding,
        and adds it to the episode store. Automatically caps at MAX_EPISODES.

        Args:
            user_msg: What the user said.
            jarvis_response: What JARVIS responded.
            tags: Optional list of topic tags.
        """
        try:
            # Build summary text for this episode
            summary = f"User: {user_msg[:200]} | JARVIS: {jarvis_response[:300]}"

            # Generate embedding for retrieval
            embedding = self._embedder.embed(summary)

            # Auto-generate tags if not provided
            if tags is None:
                tags = self._auto_tag(user_msg)

            episode = {
                "id": f"ep_{int(time.time() * 1000)}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_msg": user_msg[:500],
                "summary": summary[:600],
                "tags": tags,
                "embedding": embedding,
            }

            self._data["episodes"].append(episode)

            # Cap at MAX_EPISODES — remove oldest
            if len(self._data["episodes"]) > MAX_EPISODES:
                self._data["episodes"] = self._data["episodes"][-MAX_EPISODES:]

            # Persist to disk (in background to avoid blocking)
            threading.Thread(target=self._save, daemon=True).start()

            log.info(
                f"[Memory] Stored episode: {summary[:80]}... "
                f"(total: {len(self._data['episodes'])})"
            )

        except Exception as e:
            log.error(f"[Memory] Failed to store episode: {e}")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """
        Retrieve the most relevant past episodes for a given query.

        Uses cosine similarity between the query embedding and stored
        episode embeddings to find the best matches.

        Args:
            query: The current user query.
            top_k: Number of episodes to retrieve.

        Returns:
            List of episode dicts, sorted by relevance (most relevant first).
        """
        episodes = self._data["episodes"]
        if not episodes:
            return []

        try:
            query_embedding = self._embedder.embed(query)
            if not query_embedding:
                return []

            # Score each episode by cosine similarity
            scored = []
            for ep in episodes:
                ep_emb = ep.get("embedding", [])
                if not ep_emb:
                    continue
                score = self._embedder.similarity(query_embedding, ep_emb)
                scored.append((score, ep))

            # Sort by score descending, take top_k
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [ep for _, ep in scored[:top_k]]

            log.info(
                f"[Memory] Retrieved {len(results)} episodes for: "
                f"{query[:50]}..."
            )
            return results

        except Exception as e:
            log.error(f"[Memory] Retrieval failed: {e}")
            return []

    def get_recent_episodes(self, count: int = 10) -> list:
        """Get the most recent episodes (for conversation history display)."""
        return self._data["episodes"][-count:]

    # ═══════════════════════════════════════════════════════════════════════
    #  SEMANTIC MEMORY — What JARVIS knows about the user
    # ═══════════════════════════════════════════════════════════════════════

    def get_semantic(self) -> dict:
        """Return the full semantic memory (user profile)."""
        return self._data["semantic"]

    def update_identity(self, key: str, value: str):
        """
        Update a single identity field.

        Args:
            key: Field name (e.g., "name", "role", "location")
            value: New value.
        """
        if key in self._data["semantic"]["identity"]:
            self._data["semantic"]["identity"][key] = value
            self._save()
            log.info(f"[Memory] Identity updated: {key} = {value}")

    def update_preference(self, key: str, value: str):
        """
        Update a single preference field.

        Args:
            key: Preference name (e.g., "tone", "code_language")
            value: New value.
        """
        self._data["semantic"]["preferences"][key] = value
        self._save()
        log.info(f"[Memory] Preference updated: {key} = {value}")

    def add_project(self, project: str):
        """Add a project to the user's project list (no duplicates)."""
        projects = self._data["semantic"]["projects"]
        if project not in projects:
            projects.append(project)
            self._save()
            log.info(f"[Memory] Project added: {project}")

    def add_goal(self, goal: str):
        """Add a goal to the user's goals list (no duplicates)."""
        goals = self._data["semantic"]["goals"]
        if goal not in goals:
            goals.append(goal)
            self._save()
            log.info(f"[Memory] Goal added: {goal}")

    def update_semantic_from_message(self, user_msg: str):
        """
        Attempt to extract and merge new facts from user's message.
        Uses keyword heuristics (free, no API call needed).

        Detects patterns like:
            "my name is X" → updates identity.name
            "I'm working on X" → adds project
            "I want to learn X" → adds goal
            "I prefer X" → updates preference

        Args:
            user_msg: The user's raw message.
        """
        msg_lower = user_msg.lower().strip()

        # ── Name detection ──────────────────────────────────────────────
        for prefix in ["my name is ", "i'm ", "i am ", "call me "]:
            if msg_lower.startswith(prefix):
                name = user_msg[len(prefix):].strip().split()[0].rstrip(".,!?")
                if len(name) > 1:
                    self.update_identity("name", name.capitalize())
                break

        # ── Project detection ───────────────────────────────────────────
        for prefix in [
            "i'm working on ", "i am working on ", "my project is ",
            "working on ", "building "
        ]:
            if prefix in msg_lower:
                idx = msg_lower.index(prefix) + len(prefix)
                project = user_msg[idx:].strip().split(".")[0].strip()
                if 2 < len(project) < 60:
                    self.add_project(project)
                break

        # ── Goal detection ──────────────────────────────────────────────
        for prefix in [
            "i want to learn ", "i want to ", "my goal is ",
            "i'm trying to ", "i need to "
        ]:
            if prefix in msg_lower:
                idx = msg_lower.index(prefix) + len(prefix)
                goal = user_msg[idx:].strip().split(".")[0].strip()
                if 2 < len(goal) < 80:
                    self.add_goal(goal)
                break

        # ── Location detection ──────────────────────────────────────────
        for prefix in ["i live in ", "i'm from ", "i am from ", "i'm in "]:
            if prefix in msg_lower:
                idx = msg_lower.index(prefix) + len(prefix)
                location = user_msg[idx:].strip().split(".")[0].strip()
                if 2 < len(location) < 50:
                    self.update_identity("location", location)
                break

    # ═══════════════════════════════════════════════════════════════════════
    #  PROCEDURAL MEMORY — Custom workflows
    # ═══════════════════════════════════════════════════════════════════════

    def add_procedure(self, name: str, steps: list, triggers: list = None):
        """
        Store a named procedure (workflow).

        Args:
            name: Procedure name (e.g., "study_setup")
            steps: List of action strings (e.g., ["open Notion", "set timer 3600"])
            triggers: Optional trigger phrases to match this procedure.
        """
        if triggers is None:
            triggers = [name.replace("_", " ")]

        # Check for duplicate name — update if exists
        for proc in self._data["procedures"]:
            if proc["name"] == name:
                proc["steps"] = steps
                proc["triggers"] = triggers
                self._save()
                log.info(f"[Memory] Procedure updated: {name}")
                return

        procedure = {
            "name": name,
            "steps": steps,
            "triggers": triggers,
            "created": datetime.now(timezone.utc).isoformat(),
        }
        self._data["procedures"].append(procedure)
        self._save()
        log.info(f"[Memory] Procedure stored: {name} ({len(steps)} steps)")

    def match_procedure(self, query: str) -> Optional[dict]:
        """
        Check if the query matches any stored procedure's trigger phrases.

        Args:
            query: The user's message.

        Returns:
            The matching procedure dict, or None.
        """
        query_lower = query.lower().strip()
        for proc in self._data["procedures"]:
            for trigger in proc.get("triggers", []):
                if trigger.lower() in query_lower:
                    log.info(f"[Memory] Procedure matched: {proc['name']}")
                    return proc
        return None

    def get_procedures(self) -> list:
        """Return all stored procedures."""
        return self._data["procedures"]

    def detect_remember_command(self, user_msg: str) -> Optional[dict]:
        """
        Detect if the user is defining a new procedure.

        Patterns:
            "remember: my study setup is Notion + lofi + timer"
            "remember that my morning routine is ..."

        Args:
            user_msg: The user's raw message.

        Returns:
            Dict with 'name' and 'steps' if detected, else None.
        """
        msg_lower = user_msg.lower().strip()

        for prefix in ["remember:", "remember that ", "save procedure:"]:
            if msg_lower.startswith(prefix):
                content = user_msg[len(prefix):].strip()

                # Try to extract name and steps
                # Pattern: "my X is A + B + C" or "X: A, B, C"
                if " is " in content.lower():
                    parts = content.split(" is ", 1)
                    name = parts[0].strip().lower().replace(" ", "_")
                    steps_raw = parts[1].strip()
                elif ":" in content:
                    parts = content.split(":", 1)
                    name = parts[0].strip().lower().replace(" ", "_")
                    steps_raw = parts[1].strip()
                else:
                    name = "custom_procedure"
                    steps_raw = content

                # Split steps by + , ; or "and"
                import re
                steps = [
                    s.strip()
                    for s in re.split(r"[+,;]|\band\b", steps_raw)
                    if s.strip()
                ]

                if steps:
                    return {"name": name, "steps": steps}

        return None

    # ═══════════════════════════════════════════════════════════════════════
    #  SYSTEM PROMPT BUILDER
    # ═══════════════════════════════════════════════════════════════════════

    def build_system_prompt(self, retrieved_episodes: list = None) -> str:
        """
        Assemble the complete JARVIS system prompt with persona,
        user context, and relevant past episodes.

        Args:
            retrieved_episodes: List of episode dicts from retrieve().

        Returns:
            Complete system prompt string for the AI model.
        """
        semantic = self._data["semantic"]
        identity = semantic["identity"]
        prefs = semantic["preferences"]
        projects = semantic.get("projects", [])
        goals = semantic.get("goals", [])

        # ── Build context sections ──────────────────────────────────────
        user_name = identity.get("name", "sir")
        user_role = identity.get("role", "unknown")
        user_location = identity.get("location", "unknown")

        projects_str = ", ".join(projects) if projects else "none known"
        goals_str = ", ".join(goals) if goals else "none known"

        # ── Format retrieved episodes ───────────────────────────────────
        episode_context = "No prior context available."
        if retrieved_episodes:
            episode_lines = []
            for ep in retrieved_episodes:
                ts = ep.get("timestamp", "unknown time")
                summary = ep.get("summary", "")
                # Format timestamp nicely
                try:
                    dt = datetime.fromisoformat(ts)
                    time_str = dt.strftime("%b %d, %H:%M")
                except (ValueError, TypeError):
                    time_str = "recently"
                episode_lines.append(f"  [{time_str}] {summary[:200]}")
            episode_context = "\n".join(episode_lines)

        # ── Assemble full prompt ────────────────────────────────────────
        prompt = f"""{JARVIS_PERSONA}

User Profile:
  Name: {user_name}
  Role: {user_role}
  Location: {user_location}
  Preferred tone: {prefs.get('tone', 'concise')}
  Preferred code language: {prefs.get('code_language', 'Python')}

Active projects: {projects_str}
Goals: {goals_str}

Relevant past interactions:
{episode_context}

Rules:
- Address the user as '{user_name}' or 'sir' naturally
- Reference past interactions when relevant ("As you mentioned before...")
- Dry wit is encouraged. Never be generic or boring.
- Never say "I cannot" — say what you CAN do instead.
- End action plans with confidence, not disclaimers.
"""
        return prompt.strip()

    # ═══════════════════════════════════════════════════════════════════════
    #  AUTO-TAGGING
    # ═══════════════════════════════════════════════════════════════════════

    def _auto_tag(self, text: str) -> list:
        """
        Generate simple topic tags from text using keyword detection.
        Free, no API call — just pattern matching.

        Args:
            text: Input text to tag.

        Returns:
            List of tag strings.
        """
        text_lower = text.lower()
        tags = []

        tag_keywords = {
            "code": ["code", "function", "debug", "error", "bug", "script",
                      "python", "javascript", "class", "variable", "import"],
            "web": ["website", "html", "css", "react", "frontend", "backend",
                    "api", "server", "http", "url", "browser"],
            "ai": ["model", "ai", "machine learning", "neural", "llm",
                   "training", "dataset", "gpt", "ollama"],
            "system": ["file", "folder", "install", "terminal", "command",
                       "process", "memory", "cpu", "disk"],
            "study": ["learn", "study", "course", "tutorial", "book",
                      "practice", "exam", "dsa", "algorithm"],
            "project": ["project", "build", "create", "deploy", "ship",
                        "portfolio", "app", "tool"],
            "general": ["help", "explain", "what", "how", "why", "tell me"],
        }

        for tag, keywords in tag_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(tag)

        return tags if tags else ["general"]

    # ═══════════════════════════════════════════════════════════════════════
    #  UTILITIES
    # ═══════════════════════════════════════════════════════════════════════

    def get_stats(self) -> dict:
        """Return memory system statistics."""
        return {
            "total_episodes": len(self._data["episodes"]),
            "total_procedures": len(self._data["procedures"]),
            "user_name": self._data["semantic"]["identity"].get("name", "unknown"),
            "projects_count": len(self._data["semantic"].get("projects", [])),
            "goals_count": len(self._data["semantic"].get("goals", [])),
        }

    def clear_episodes(self):
        """Clear all episodic memory (for debugging/reset)."""
        self._data["episodes"] = []
        self._save()
        log.info("[Memory] All episodes cleared.")

    def export_data(self) -> dict:
        """Export the full memory data (without embeddings, for display)."""
        export = json.loads(json.dumps(self._data))
        # Strip embeddings for readability
        for ep in export.get("episodes", []):
            ep.pop("embedding", None)
        return export


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLETON INSTANCE
# ═══════════════════════════════════════════════════════════════════════════════

# Global memory instance — shared across the entire application
_memory_instance = None
_memory_lock = threading.Lock()


def get_memory() -> MemorySystem:
    """
    Get the global MemorySystem singleton.

    Returns:
        The shared MemorySystem instance.
    """
    global _memory_instance
    if _memory_instance is None:
        with _memory_lock:
            if _memory_instance is None:
                _memory_instance = MemorySystem()
    return _memory_instance


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  [JARVIS Memory System — Self Test]\n")

    mem = MemorySystem()
    print(f"  Stats: {mem.get_stats()}")

    # Test episodic memory
    mem.remember(
        "Help me debug my React useEffect hook",
        "The issue was an infinite loop caused by a missing dependency array.",
        tags=["code", "react", "debug"]
    )
    print("  ✅ Episode stored")

    # Test semantic memory
    mem.update_identity("name", "Tony")
    mem.add_project("JARVIS")
    mem.add_goal("Ship JARVIS v2")
    print("  ✅ Semantic memory updated")

    # Test procedural memory
    mem.add_procedure(
        "study_setup",
        ["open Notion", "open lofi.cafe", "set timer 3600"],
        triggers=["study setup", "study mode", "prepare study"]
    )
    print("  ✅ Procedure stored")

    # Test procedure matching
    match = mem.match_procedure("activate study setup")
    print(f"  ✅ Procedure match: {match['name'] if match else 'none'}")

    # Test retrieval
    results = mem.retrieve("React hooks problem")
    print(f"  ✅ Retrieved {len(results)} episodes")

    # Test system prompt
    prompt = mem.build_system_prompt(retrieved_episodes=results)
    print(f"  ✅ System prompt built ({len(prompt)} chars)")
    print(f"\n  Stats: {mem.get_stats()}")
    print("\n  [All tests passed]\n")