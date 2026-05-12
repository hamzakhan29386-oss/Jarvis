"""Compatibility wrapper for the production wake listener in wake.service."""

from wake.service import ProductionWakeService, get_wake_service

__all__ = ["ProductionWakeService", "get_wake_service"]

