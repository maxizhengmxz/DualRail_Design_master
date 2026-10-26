"""Lightweight process-runner test handler; no physics imports."""


def diagnostic_task(payload):
    """Deterministic adapter used only by runner regression tests."""
    if payload.get("raise_error"):
        raise ValueError("intentional test failure")
    return {"sum": sum(payload.get("values", []))}
