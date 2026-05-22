"""v7 acquisition function: fast heuristic, no model calls.

The acquisition function is called ~256K times (once per candidate pair in the
hidden pool), so model inference is not feasible. This heuristic scores pairs by
estimated prediction uncertainty using only string-parsed metadata:

  1. Subject uncertainty: intermediate-sized models (~10B params) are hardest to
     predict — large models almost always pass, tiny models almost always fail.
     Score peaks at log10(params) ≈ 1 (10B) and falls off toward extremes.

  2. Item diversity: a hash of benchmark + item_content spreads labels across
     different topics, avoiding K labels all from the same narrow item cluster.

No imports beyond stdlib are needed — this runs in microseconds per call.
"""

from __future__ import annotations

import math


def _parse_log_params(subject_content: str) -> float | None:
    """Return log10(params in billions) from subject_content, or None if unparseable."""
    for line in subject_content.splitlines():
        if line.startswith("Parameters:"):
            val = line[11:].strip().upper()
            try:
                if val.endswith("T"):
                    return math.log10(float(val[:-1]) * 1000)
                if val.endswith("B"):
                    return math.log10(max(float(val[:-1]), 1e-3))
                if val.endswith("M"):
                    return math.log10(max(float(val[:-1]) / 1000, 1e-6))
            except ValueError:
                pass
    return None


def acquisition_function(input: dict) -> float:
    """Score a candidate pair by estimated labeling value. Higher = more desired."""
    # --- Subject uncertainty ---
    # Prediction is most uncertain around ~10B params (log10 ≈ 1.0).
    # Penalise extremes (tiny models always fail, huge models always pass).
    log_p = _parse_log_params(input.get("subject_content", ""))
    if log_p is not None:
        # Gaussian-like score peaked at 10B; std ≈ 1 order of magnitude
        subject_score = math.exp(-0.5 * ((log_p - 1.0) ** 2))
    else:
        subject_score = 0.5   # unknown size → moderate prior

    # --- Item diversity (hash-based) ---
    # Spread labels across different item clusters within each benchmark.
    # Use a simple polynomial hash of benchmark + item_content.
    text = input.get("benchmark", "") + "|" + input.get("item_content", "")
    h = 0
    for ch in text[:200]:   # first 200 chars; fast
        h = (h * 31 + ord(ch)) & 0xFFFF
    diversity_score = (h % 1000) / 1000.0   # in [0, 1)

    return subject_score + 0.2 * diversity_score
