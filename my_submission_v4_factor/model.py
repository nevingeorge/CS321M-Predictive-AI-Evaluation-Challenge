"""PGE (Prediction-Guided Evaluation) submission — K=4 factor model + neural heads.

Architecture:
  P(Y_ij=1) = sigmoid(U_i^T V_hat_j + Z_hat_j)

  Item branch  (Stage 2a):
    E_j = sentence_encoder("Benchmark: ...\nCondition: ...\nItem: ...")
    [V_hat_j, Z_hat_j] = Linear(256, K+1)(ReLU(Linear(768, 256)(E_j)))

  Model branch (Stage 2b):
    Primary: look up U_i from subject_abilities.npy by display name
    Fallback: U_hat_i = Linear(d_F, K)(structured_features(subject_content))

Module-level code runs once at container start.
predict() is called once per (model_id, item_id) pair.
"""

from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Smoke-test helpers (shared pattern with other submissions)
# ---------------------------------------------------------------------------

LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
K = 4
REFERENCE_DATE = datetime(2020, 1, 1)

# HuggingFace repo where training artifacts are stored.
# Must match the second line of models.txt so the platform pre-downloads it.
ARTIFACTS_REPO = "ngeorge/cs321m-competition"


def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _declared_models() -> list[str]:
    models_path = Path(__file__).with_name("models.txt")
    if not models_path.exists():
        return []
    return [
        line.strip()
        for line in models_path.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _resolve_cache_dir() -> str | None:
    candidates = [
        os.environ.get("HF_HOME", "").strip(),
        "/app/hf_cache",
        str(Path(__file__).with_name(".hf_cache")),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        if os.access(path, os.W_OK):
            return str(path)
    return None


# ---------------------------------------------------------------------------
# Subject metadata parsing (for model branch fallback)
# ---------------------------------------------------------------------------

def _parse_name(subject_content: str) -> str:
    for line in subject_content.splitlines():
        if line.startswith("Name:"):
            return line[5:].strip()
    return subject_content.strip()


def _parse_params_log(params_str: str) -> float:
    if not params_str:
        return 0.0
    s = params_str.strip().upper()
    try:
        if s.endswith("T"):
            return float(math.log10(float(s[:-1]) * 1000))
        if s.endswith("B"):
            return float(math.log10(max(float(s[:-1]), 1e-3)))
        if s.endswith("M"):
            return float(math.log10(max(float(s[:-1]) / 1000, 1e-6)))
        return float(math.log10(max(float(s), 1e-3)))
    except ValueError:
        return 0.0


def _parse_release_days(date_str: str) -> float:
    if not date_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = datetime.strptime(date_str.strip(), fmt)
            return float((d - REFERENCE_DATE).days)
        except ValueError:
            continue
    return 0.0


def _parse_subject_content(subject_content: str) -> dict[str, str]:
    result = {}
    for line in subject_content.splitlines():
        for key, label in [("name", "Name"), ("organization", "Organization"),
                           ("params", "Parameters"), ("release_date", "Released"),
                           ("family", "Family")]:
            if line.startswith(f"{label}:"):
                result[key] = line[len(label) + 1:].strip()
    return result


def _make_onehot(value: str, vocab: dict[str, int]) -> list[float]:
    idx = vocab.get(value, vocab.get("<other>", 0))
    vec = [0.0] * len(vocab)
    if 0 <= idx < len(vec):
        vec[idx] = 1.0
    return vec


def _subject_to_features(subject_content: str) -> list[float] | None:
    if FEATURE_CONFIG is None:
        return None
    meta = _parse_subject_content(subject_content)
    log_p = _parse_params_log(meta.get("params", ""))
    rel_d = _parse_release_days(meta.get("release_date", ""))
    fam_oh = _make_onehot(meta.get("family", ""), FEATURE_CONFIG["family_vocab"])
    org_oh = _make_onehot(meta.get("organization", ""), FEATURE_CONFIG["org_vocab"])
    return [log_p, rel_d] + fam_oh + org_oh


# ---------------------------------------------------------------------------
# Module-level init
# ---------------------------------------------------------------------------

ENCODER = None
ITEM_HEAD = None
MODEL_HEAD = None
SUBJECT_U = None
SUBJECT_INDEX: dict[str, int] | None = None
FEATURE_CONFIG: dict | None = None

if _local_smoke_test_enabled():
    print("[model_v4] Skipped loading for local smoke test.", flush=True)
else:
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except Exception as exc:
        raise RuntimeError(
            "torch, numpy, and sentence_transformers are required."
        ) from exc

    try:
        _cache = _resolve_cache_dir()
        _here = Path(__file__).parent   # artifact files are bundled in the ZIP

        ENCODER = SentenceTransformer(
            "sentence-transformers/all-mpnet-base-v2",
            cache_folder=_cache,
            local_files_only=True,
        )

        _weights = torch.load(_here / "nn_weights.pt", map_location="cpu")
        FEATURE_CONFIG = _weights["feature_config"]
        _d_F = FEATURE_CONFIG["d_F"]

        ITEM_HEAD = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, K + 1),
        )
        ITEM_HEAD.load_state_dict(_weights["item_head"])
        ITEM_HEAD.eval()

        MODEL_HEAD = nn.Linear(_d_F, K, bias=True)
        MODEL_HEAD.load_state_dict(_weights["model_head"])
        MODEL_HEAD.eval()

        SUBJECT_U = np.load(_here / "subject_abilities.npy")
        SUBJECT_INDEX = json.loads((_here / "subject_index.json").read_text())
        print(f"[model_v4] Loaded. {len(SUBJECT_INDEX)} known subjects.", flush=True)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load PGE model artifacts: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Prediction cache
# ---------------------------------------------------------------------------

_CACHE: dict[str, float] = {}


def _cache_key(input: dict) -> str:
    return json.dumps(
        {k: input[k] for k in ("benchmark", "condition", "subject_content", "item_content")},
        sort_keys=True,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly)."""
    if ENCODER is None:
        return 0.5

    import torch
    import numpy as np

    key = _cache_key(input)
    if not labeled and key in _CACHE:
        return _CACHE[key]

    # Item branch: sentence encoder → MLP → (V_hat [K], Z_hat [1])
    item_text = (
        f"Benchmark: {input['benchmark']}\n"
        f"Condition: {input['condition']}\n"
        f"Item: {input['item_content']}"
    )
    enc_item = torch.tensor(ENCODER.encode(item_text), dtype=torch.float32)
    with torch.no_grad():
        out = ITEM_HEAD(enc_item)   # [K+1]
    v_hat = out[:K]                 # [K]
    z_hat = out[K]                  # scalar

    # Model branch: lookup table (primary) or linear predictor (fallback)
    name = _parse_name(input["subject_content"])
    if SUBJECT_INDEX is not None and name in SUBJECT_INDEX:
        u = torch.tensor(SUBJECT_U[SUBJECT_INDEX[name]], dtype=torch.float32)
    else:
        feats = _subject_to_features(input["subject_content"])
        if feats is not None and MODEL_HEAD is not None:
            f_i = torch.tensor(feats, dtype=torch.float32)
            with torch.no_grad():
                u = MODEL_HEAD(f_i)
        else:
            u = torch.zeros(K)

    logit = (u * v_hat).sum() + z_hat
    p = float(torch.sigmoid(logit).item())

    if not labeled:
        _CACHE[key] = p
    return p
