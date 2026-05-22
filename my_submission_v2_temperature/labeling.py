"""Uncertainty-based adaptive labeling for the Predictive AI Evaluation Challenge.

Scores each candidate by prediction entropy: inputs where the judge is most
uncertain (predicted probability ≈ 0.5) are prioritised for labeling.
Predictions are shared with model.py via its module-level cache, so acquisition
does not double the inference cost when predict() is called later with labeled=None.
"""

from __future__ import annotations

import model as _model


def acquisition_function(input: dict) -> float:
    """Return a labeling-priority score. Higher = more desired.

    Score = -|p - 0.5|, so p=0.5 (maximum uncertainty) → score=0 (highest),
    and p=0 or p=1 (confident) → score=-0.5 (lowest).
    """
    p = _model.predict(input, labeled=None)
    return -abs(p - 0.5)
