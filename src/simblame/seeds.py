"""Seed specifications: ``50`` (0..49), ``10:20`` (10..19), ``1,5,9``, or combinations.

Dependency-free, so every layer can use it (the CLI parses ``--seeds`` with it,
the per-version worker parses its explicit seed list).
"""

from __future__ import annotations


def parse_seeds(spec) -> list:
    """``"50"`` -> 0..49, ``"10:20"`` -> 10..19, ``"1,5,9"`` -> [1, 5, 9]; ValueError if invalid."""
    text = str(spec).strip()
    if text.isdigit():
        if int(text) == 0:
            raise ValueError("need at least one seed")
        return list(range(int(text)))
    seeds: list = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition(":")
        try:
            if sep:
                if int(hi) <= int(lo):
                    raise ValueError(f"empty range {part!r} (use lo:hi with hi > lo)")
                seeds.extend(range(int(lo), int(hi)))
            else:
                seeds.append(int(part))
        except ValueError as e:
            if "empty range" in str(e):
                raise
            raise ValueError(f"not a seed or range: {part!r} (use N, lo:hi or 1,5,9)") from None
    if not seeds:
        raise ValueError(f"no seeds in {spec!r}")
    return seeds


def parse_seed_list(text: str) -> list:
    """Explicit comma-separated seeds (``"7"`` is the single seed 7, not a count)."""
    return [int(x) for x in str(text).split(",") if x.strip()]
