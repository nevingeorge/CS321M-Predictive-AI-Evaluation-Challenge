"""v11 acquisition: prefer pairs where the lookup is most uncertain.

The lookup returns empirical accuracy (a real fraction). The most uncertain
pairs are those closest to 0.5. No model inference needed.
"""

from __future__ import annotations

import model as _model


def acquisition_function(input: dict) -> float:
    """Return -|p - 0.5|. Higher = more uncertain = more valuable to label."""
    p = _model.predict(input, labeled=None)
    return -abs(p - 0.5)
