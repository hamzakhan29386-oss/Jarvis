"""
main.py — JARVIS Entry Point
==============================
Fast, local AI assistant powered by Ollama.

Run with:
    python main.py

Requirements:
    pip install requests
    Ollama must be running: ollama serve
"""

import sys
from brain import check_ollama, stream_response

# ── Banner ──────────────────────────────────────────────────────────────────
BANNER = r"""
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗         ║
║          ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝         ║
║          ██║███████║██████╔╝██║   ██║██║███████╗         ║
║     ██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║         ║
║     ╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║         ║
║      ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝         ║
║                                                          ║
║        Your Local AI Assistant  ·  Powered by Ollama     ║
║                  Model: phi3  ·  Fast Mode                ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""

# Words that trigger exit
EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "stop", "shutdown", "shut down"}


def main():
    """Main JARVIS loop: Ask → Stream → Repeat."""

    print(BANNER)

    # ── Check Ollama is running ─────────────────────────────────────────
    print("  ⏳  Checking Ollama connection...")
    if check_ollama():
        print("  🟢  Ollama is running. JARVIS is ready!\n")
    else:
        print("  ❌  Cannot reach Ollama at localhost:11434")
        print("  💡  Start it with:  ollama serve")
        print("  💡  Then pull the model:  ollama pull phi3")
        print("  💡  Then run this script again.\n")
        print("  Continuing anyway (will retry on each message)...\n")

    print("  💡  Type your message and press Enter.")
    print("  💡  Type 'exit' to quit.\n")
    print("  " + "═" * 56 + "\n")

    # ── Main loop ───────────────────────────────────────────────────────
    while True:
        try:
            # Get user input
            try:
                user_input = input("  You ➤  ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n  👋  Goodbye!")
                break

            # Skip empty input
            if not user_input:
                continue

            # Check for exit
            if user_input.lower() in EXIT_WORDS:
                print("\n  🤖  JARVIS ➤  Goodbye! Shutting down.\n")
                break

            # Stream AI response
            sys.stdout.write("\n  🤖  JARVIS ➤  ")
            sys.stdout.flush()

            response = stream_response(user_input)

            # Separator
            print("\n  " + "─" * 56 + "\n")

        except KeyboardInterrupt:
            print("\n\n  👋  Interrupted. Shutting down JARVIS...")
            break
        except Exception as e:
            print(f"\n  ❌  Error: {e}")
            print("  🔄  Recovering...\n")
            continue


if __name__ == "__main__":
    main()
