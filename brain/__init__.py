"""Cerveau's engine: pure Python, no Django and no vendor SDK imports.

The engine reaches storage and LLMs only through the Protocols in `brain.ports`;
implementations live outside, in `adapters/` (pure Python) and `apps/cerveau/`
(Django).
"""
