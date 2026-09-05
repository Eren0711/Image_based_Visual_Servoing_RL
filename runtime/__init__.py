"""Shared construction, seeding, and provenance utilities for new runs.

Imports stay explicit (for example ``runtime.environment``) so low-level
wrappers can use ``runtime.seeding`` without creating package import cycles.
"""
