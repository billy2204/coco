"""Gói src. Ép stdout/stderr sang UTF-8 để in tiếng Việt được trên console Windows (cp1252)."""
import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover
            pass
