"""Uncertainty-based adaptive labeling — shared with other submissions."""

from __future__ import annotations

import model as _model


def acquisition_function(input: dict) -> float:
    """Return a labeling-priority score. Higher = more desired.

    Score = -|p - 0.5|: most uncertain inputs (p ≈ 0.5) get the highest score.
    """
    p = _model.predict(input, labeled=None)
    return -abs(p - 0.5)
