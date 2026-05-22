"""v5 acquisition function: prioritise items where factor model and LLM judge disagree most.

High disagreement = high uncertainty about the true label = most valuable to reveal.
"""

from __future__ import annotations

import model as _model


def acquisition_function(input: dict) -> float:
    """Return |p_factor - p_llm|. Higher = more desired for labeling."""
    p_factor = _model._predict_factor(input)
    p_llm = _model._predict_llm(input)
    return abs(p_factor - p_llm)
