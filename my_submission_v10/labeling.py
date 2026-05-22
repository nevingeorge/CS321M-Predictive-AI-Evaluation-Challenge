"""v10 acquisition function: model disagreement (Query by Committee).

Same strategy as v5: select pairs where the factor model and LLM judge disagree
most. High disagreement = high uncertainty = most informative label to reveal.

Side benefit: calling _predict_factor (which calls _get_logits) during acquisition
pre-populates _LOGIT_CACHE for all candidate pairs, so predict() hits the cache
for every scored pair and runs near-instantly.
"""

from __future__ import annotations

import model as _model


def acquisition_function(input: dict) -> float:
    """Return |p_factor - p_llm|. Higher = more desired for labeling."""
    p_factor = _model._predict_factor(input)
    p_llm = _model._predict_llm(input)
    return abs(p_factor - p_llm)
