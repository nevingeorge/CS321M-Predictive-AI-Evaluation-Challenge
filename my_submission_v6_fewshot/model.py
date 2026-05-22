"""v5 Ensemble: K=4 factor model + LLM-as-judge, with Platt scaling.

Prediction pipeline:
  logit_factor  = U_i^T V_hat_j + Z_hat_j          (factor model branch)
  logit_llm     = log P(yes) - log P(no)            (LLM branch, Qwen3.5-9B)
  logit_ens     = w_factor*logit_factor + w_llm*logit_llm + bias  (tuned offline)

  If labeled examples are provided → fit Platt scaling (a, b) via gradient descent:
    p = sigmoid(a * logit_ens + b)
  Otherwise:
    p = sigmoid(logit_ens)

Logits are cached per input so acquisition_function() and predict() share work.
Module-level code runs once; predict() is called once per (model_id, item_id) pair.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"
K = 4
ENCODER_DIM = 1024          # BAAI/bge-large-en-v1.5 output dimension
REFERENCE_DATE = datetime(2020, 1, 1)
ARTIFACTS_REPO = "ngeorge/cs321m-competition"

JUDGE_TEMPLATE = (
    "You will see a description of an AI subject and an evaluation item. "
    "Decide whether the subject would answer the item correctly. "
    "Reply with a single token: yes or no.\n\n"
    "Benchmark: {benchmark}\n"
    "Condition: {condition}\n"
    "Subject: {subject_content}\n"
    "Item: {item_content}\n"
    "Answer:"
)

# ---------------------------------------------------------------------------
# Smoke-test / cache helpers
# ---------------------------------------------------------------------------

def _local_smoke_test_enabled() -> bool:
    value = os.environ.get(LOCAL_SMOKE_TEST_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
# Subject metadata helpers
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


def _is_instruct(meta: dict) -> float:
    text = (meta.get("name", "") + " " + meta.get("family", "")).lower()
    return 1.0 if any(k in text for k in ["instruct", "chat", "-it", "rlhf"]) else 0.0


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
    instruct = _is_instruct(meta)
    fam_oh = _make_onehot(meta.get("family", ""), FEATURE_CONFIG["family_vocab"])
    org_oh = _make_onehot(meta.get("organization", ""), FEATURE_CONFIG["org_vocab"])
    return [log_p, rel_d, instruct] + fam_oh + org_oh


# ---------------------------------------------------------------------------
# Module-level globals
# ---------------------------------------------------------------------------

ENCODER = None          # SentenceTransformer (bge-large-en-v1.5)
ITEM_HEAD = None        # nn.Sequential(Linear(1024,256), ReLU, Linear(256, K+1))
MODEL_HEAD = None       # nn.Linear(d_F, K)
SUBJECT_U = None        # np.ndarray [n_subjects, K]
SUBJECT_INDEX = None    # dict[str, int]
FEATURE_CONFIG = None   # family/org vocabs + d_F

LLM = None              # AutoModelForCausalLM (Qwen3.5-9B)
LLM_TOKENIZER = None    # AutoTokenizer
YES_ID: int | None = None
NO_ID: int | None = None

W_FACTOR = 1.0
W_LLM = 1.0
W_BIAS = 0.0

# Per-call caches (keyed by json of input fields)
_LOGIT_CACHE: dict[str, tuple[float, float]] = {}   # → (logit_factor, logit_llm) zero-shot
_FEW_SHOT_LLM_CACHE: dict[str, float] = {}          # → logit_llm with few-shot context
_platt_cache: dict = {"key": None, "a": 1.0, "b": 0.0}

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

if _local_smoke_test_enabled():
    print("[model_v5] Skipped loading for local smoke test.", flush=True)
else:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Missing dependency.") from exc

    try:
        _cache = _resolve_cache_dir()
        _here = Path(__file__).parent   # small artifact files bundled in the ZIP

        # Factor model
        ENCODER = SentenceTransformer(
            "BAAI/bge-large-en-v1.5", cache_folder=_cache, local_files_only=True
        )

        _weights = torch.load(_here / "nn_weights.pt", map_location="cpu")
        FEATURE_CONFIG = _weights["feature_config"]
        _d_F = FEATURE_CONFIG["d_F"]

        ITEM_HEAD = nn.Sequential(
            nn.Linear(ENCODER_DIM, 256), nn.ReLU(), nn.Linear(256, K + 1)
        )
        ITEM_HEAD.load_state_dict(_weights["item_head"])
        ITEM_HEAD.eval()

        MODEL_HEAD = nn.Linear(_d_F, K, bias=True)
        MODEL_HEAD.load_state_dict(_weights["model_head"])
        MODEL_HEAD.eval()

        SUBJECT_U = np.load(_here / "subject_abilities.npy")
        SUBJECT_INDEX = json.loads((_here / "subject_index.json").read_text())

        # Ensemble weights
        _ew = json.loads((_here / "ensemble_weights.json").read_text())
        W_FACTOR = float(_ew["w_factor"])
        W_LLM = float(_ew["w_llm"])
        W_BIAS = float(_ew["bias"])

        # LLM judge (declared in models.txt, pre-downloaded by platform)
        LLM_TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen3.5-9B", cache_dir=_cache, local_files_only=True
        )
        LLM = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16,
            device_map="auto", cache_dir=_cache, local_files_only=True,
        )
        LLM.eval()
        YES_ID = LLM_TOKENIZER.encode("yes", add_special_tokens=False)[-1]
        NO_ID = LLM_TOKENIZER.encode("no", add_special_tokens=False)[-1]

        print(
            f"[model_v5] Loaded. {len(SUBJECT_INDEX)} subjects. "
            f"w_factor={W_FACTOR:.3f} w_llm={W_LLM:.3f} bias={W_BIAS:.3f}",
            flush=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load v5 artifacts: {exc}") from exc


# ---------------------------------------------------------------------------
# Logit computation
# ---------------------------------------------------------------------------

def _cache_key(input: dict) -> str:
    return json.dumps(
        {k: input[k] for k in ("benchmark", "condition", "subject_content", "item_content")},
        sort_keys=True,
    )


def _compute_factor_logit(input: dict) -> float:
    import torch
    item_text = (
        f"Benchmark: {input['benchmark']}\n"
        f"Condition: {input['condition']}\n"
        f"Item: {input['item_content']}"
    )
    enc = torch.tensor(ENCODER.encode(item_text), dtype=torch.float32)
    with torch.no_grad():
        out = ITEM_HEAD(enc)
    v_hat, z_hat = out[:K], out[K]

    name = _parse_name(input["subject_content"])
    if SUBJECT_INDEX and name in SUBJECT_INDEX:
        u = torch.tensor(SUBJECT_U[SUBJECT_INDEX[name]], dtype=torch.float32)
    else:
        feats = _subject_to_features(input["subject_content"])
        if feats is not None:
            with torch.no_grad():
                u = MODEL_HEAD(torch.tensor(feats, dtype=torch.float32))
        else:
            u = torch.zeros(K)
    return float(((u * v_hat).sum() + z_hat).item())


def _build_judge_prompt(input: dict, labeled: list[dict] | None = None) -> str:
    prefix = ""
    if labeled:
        same = [ex for ex in labeled if ex.get("benchmark") == input.get("benchmark")]
        for ex in (same[:4] if same else labeled[:4]):
            ans = "yes" if ex["label"] == 1 else "no"
            prefix += JUDGE_TEMPLATE.format(**ex) + f" {ans}\n\n"
    return prefix + JUDGE_TEMPLATE.format(**input)


def _run_llm(prompt: str) -> float:
    import torch
    ids = LLM_TOKENIZER(prompt, return_tensors="pt").to(LLM.device)
    with torch.no_grad():
        logits = LLM(**ids).logits[0, -1]
    lp = torch.log_softmax(logits, dim=-1)
    return float(lp[YES_ID].item() - lp[NO_ID].item())


def _compute_llm_logit(input: dict) -> float:
    return _run_llm(_build_judge_prompt(input))


def _compute_llm_logit_fewshot(input: dict, labeled: list[dict]) -> float:
    """LLM logit with labeled examples as few-shot context. Cached per (input, labeled set)."""
    key = _cache_key(input) + "|" + str(_labeled_key(labeled))
    if key in _FEW_SHOT_LLM_CACHE:
        return _FEW_SHOT_LLM_CACHE[key]
    ll = _run_llm(_build_judge_prompt(input, labeled))
    _FEW_SHOT_LLM_CACHE[key] = ll
    return ll


def _get_logits(input: dict) -> tuple[float, float]:
    """Return (logit_factor, logit_llm), computing and caching on first call."""
    key = _cache_key(input)
    if key in _LOGIT_CACHE:
        return _LOGIT_CACHE[key]
    lf = _compute_factor_logit(input)
    ll = _compute_llm_logit(input)
    _LOGIT_CACHE[key] = (lf, ll)
    return lf, ll


def _ensemble_logit(input: dict) -> float:
    lf, ll = _get_logits(input)
    return W_FACTOR * lf + W_LLM * ll + W_BIAS


# ---------------------------------------------------------------------------
# Platt scaling (fitted online from labeled examples)
# ---------------------------------------------------------------------------

def _fit_platt(logits, labels, steps: int = 400, lr: float = 0.05):
    """Gradient descent on sigmoid(a*logit + b). Returns (a, b)."""
    import numpy as np
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    a, b = 1.0, 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        err = p - labels
        a -= lr * float(np.mean(err * logits))
        b -= lr * float(np.mean(err))
    return float(a), float(b)


def _labeled_key(labeled: list[dict]) -> tuple:
    return tuple(
        (ex.get("item_content", "")[:40], ex.get("label")) for ex in labeled
    )


def _ensemble_logit_fewshot(input: dict, labeled: list[dict]) -> float:
    """Ensemble logit using few-shot LLM logit + cached factor logit."""
    lf, _ = _get_logits(input)
    ll = _compute_llm_logit_fewshot(input, labeled)
    return W_FACTOR * lf + W_LLM * ll + W_BIAS


def _get_platt_params(labeled: list[dict]) -> tuple[float, float]:
    """Fit Platt scaling using few-shot ensemble logits for consistency with predict()."""
    key = _labeled_key(labeled)
    if key == _platt_cache["key"]:
        return _platt_cache["a"], _platt_cache["b"]
    ens_logits = [_ensemble_logit_fewshot(ex, labeled) for ex in labeled]
    labels = [float(ex["label"]) for ex in labeled]
    if len(set(labels)) < 2:
        return 1.0, 0.0
    a, b = _fit_platt(ens_logits, labels)
    _platt_cache.update({"key": key, "a": a, "b": b})
    return a, b


# ---------------------------------------------------------------------------
# Semi-public helpers for labeling.py
# ---------------------------------------------------------------------------

def _predict_factor(input: dict) -> float:
    if ENCODER is None:
        return 0.5
    lf, _ = _get_logits(input)   # populates cache for both logits
    return float(1.0 / (1.0 + math.exp(-lf)))


def _predict_llm(input: dict) -> float:
    if LLM is None:
        return 0.5
    _, ll = _get_logits(input)   # hits cache if _predict_factor already ran
    return float(1.0 / (1.0 + math.exp(-ll)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly)."""
    if ENCODER is None and LLM is None:
        return 0.5

    if labeled:
        # Few-shot path: LLM sees labeled examples as context, Platt fitted consistently
        logit_ens = _ensemble_logit_fewshot(input, labeled)
        a, b = _get_platt_params(labeled)
        logit_ens = a * logit_ens + b
    else:
        # Zero-shot path: use cached logits
        logit_ens = _ensemble_logit(input)

    return float(1.0 / (1.0 + math.exp(-logit_ens)))
