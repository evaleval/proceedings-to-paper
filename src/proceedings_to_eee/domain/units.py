"""Conservative canonicalization for explicitly reported measurement units."""

from __future__ import annotations

import re

_UNIT_ALIASES = {
    "%": "percent",
    "pct": "percent",
    "percent": "percent",
    "percentage": "percent",
    "percentages": "percent",
    "percentage point": "percentage_points",
    "percentage points": "percentage_points",
    "proportion": "proportion",
    "proportions": "proportion",
    "probability": "probability",
    "probabilities": "probability",
}


def canonicalize_unit(unit: str | None) -> str | None:
    """Return a canonical alias without changing the reported numeric scale.

    Only explicit lexical aliases are normalized. In particular, ``%`` becomes
    ``percent`` while its associated numeric value remains untouched. Unknown
    units are preserved (apart from surrounding whitespace) instead of guessed.
    """

    if unit is None:
        return None
    stripped = unit.strip()
    normalized = re.sub(r"[\s_-]+", " ", stripped.casefold())
    return _UNIT_ALIASES.get(normalized, stripped)
