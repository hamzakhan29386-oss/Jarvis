"""TTS facade for the native voice package."""

from __future__ import annotations


def speak(text: str) -> None:
    from voice import get_voice_engine

    get_voice_engine().speak_async(text)
