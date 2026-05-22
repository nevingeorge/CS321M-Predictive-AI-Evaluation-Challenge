"""v11: Per-(subject, benchmark, condition) average accuracy lookup.

No HuggingFace models, no GPU, no network calls. Predictions come directly
from the empirical correctness rates in the public training data.

Lookup priority:
  1. (subject, benchmark, condition) — most specific
  2. (subject, benchmark)            — condition-agnostic fallback
  3. (subject,)                      — subject overall accuracy
  4. (benchmark,)                    — benchmark average (unknown subject)
  5. global_mean                     — final fallback

Labeled examples (when provided) are used to calibrate predictions for
the current round's benchmark/condition, blended with the lookup prior.
"""

from __future__ import annotations

import json
from pathlib import Path

_here = Path(__file__).parent

LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"


def _local_smoke_test_enabled() -> bool:
    import os
    return os.environ.get(LOCAL_SMOKE_TEST_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


# Load lookup table at module init (JSON, no models)
LOOKUP: dict = {}
GLOBAL_MEAN: float = 0.5

if _local_smoke_test_enabled():
    print("[model_v11] Using empty lookup for smoke test.", flush=True)
else:
    _raw = json.loads((_here / "lookup.json").read_text())
    LOOKUP = _raw
    GLOBAL_MEAN = float(_raw.get("global_mean", 0.5))
    print(
        f"[model_v11] Loaded lookup: "
        f"{len(_raw.get('subjects', {}))} subjects, "
        f"{len(_raw.get('benchmarks', {}))} benchmarks.",
        flush=True,
    )


def _parse_name(subject_content: str) -> str:
    for line in subject_content.splitlines():
        if line.startswith("Name:"):
            return line[5:].strip()
    return subject_content.strip()


def _lookup_base(name: str, benchmark: str, condition: str) -> float:
    """Look up empirical accuracy with priority fallback chain."""
    subj = LOOKUP.get("subjects", {}).get(name)
    if subj:
        # (benchmark, condition)
        v = subj.get(f"{benchmark}|{condition}")
        if v is not None:
            return float(v)
        # (benchmark only)
        v = subj.get(benchmark)
        if v is not None:
            return float(v)
        # subject overall
        v = subj.get("__overall__")
        if v is not None:
            return float(v)

    # benchmark average (unknown subject)
    v = LOOKUP.get("benchmarks", {}).get(benchmark)
    if v is not None:
        return float(v)

    return GLOBAL_MEAN


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly)."""
    name      = _parse_name(input.get("subject_content", ""))
    benchmark = input.get("benchmark", "")
    condition = input.get("condition", "none")

    base = _lookup_base(name, benchmark, condition)

    # Use labeled examples from the same benchmark to calibrate.
    # Blend: 70% labeled average + 30% lookup prior (if ≥2 labeled examples).
    if labeled:
        same = [ex for ex in labeled
                if ex.get("benchmark") == benchmark and "label" in ex]
        if len(same) >= 2:
            labeled_mean = sum(ex["label"] for ex in same) / len(same)
            base = 0.7 * labeled_mean + 0.3 * base

    return float(base)
