"""Parsing of private stance-probe responses.

Malformed responses are never discarded: the raw text is kept, parsing is
attempted deterministically, an optional repair call can be made, and the
outcome is recorded explicitly in `parse_status`:

    ok          valid JSON object with all keys
    extracted   JSON object found inside surrounding text / code fences
    regex       values recovered with a key: number pattern
    clamped     (suffix) a value was outside the scale and was clamped
    repaired    recovered by a separate repair call
    failed      nothing usable; values are NaN
"""

from __future__ import annotations

import json
import math
import re

from ..config.schema import StanceDimension


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def _coerce(v) -> float | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, str):
        m = re.search(r"-?\d+(?:\.\d+)?", v)
        return float(m.group(0)) if m else None
    return None


def _validate(obj, dims: dict[str, StanceDimension]) -> tuple[dict[str, float] | None, bool]:
    if not isinstance(obj, dict):
        return None, False
    out, clamped = {}, False
    for name, d in dims.items():
        v = _coerce(obj.get(name))
        if v is None:
            return None, False
        if v < d.min or v > d.max:
            v = min(max(v, d.min), d.max)
            clamped = True
        out[name] = v
    return out, clamped


def parse_probe(text: str, dims: dict[str, StanceDimension]) -> tuple[dict[str, float] | None, str]:
    stripped = text.strip()
    try:
        vals, clamped = _validate(json.loads(stripped), dims)
        if vals is not None:
            return vals, "ok+clamped" if clamped else "ok"
    except (json.JSONDecodeError, ValueError):
        pass
    candidate = _first_json_object(stripped)
    if candidate is not None:
        try:
            vals, clamped = _validate(json.loads(candidate), dims)
            if vals is not None:
                return vals, "extracted+clamped" if clamped else "extracted"
        except (json.JSONDecodeError, ValueError):
            pass
    found = {}
    for name in dims:
        m = re.search(rf'["\']?{re.escape(name)}["\']?\s*[:=]\s*"?(-?\d+(?:\.\d+)?)', stripped)
        if m:
            found[name] = float(m.group(1))
    vals, clamped = _validate(found, dims)
    if vals is not None:
        return vals, "regex+clamped" if clamped else "regex"
    return None, "failed"
