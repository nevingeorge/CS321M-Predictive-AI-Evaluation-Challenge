"""LLM-as-judge submission for the Predictive AI Evaluation Challenge.

Loads Qwen/Qwen3.5-9B from the local HF cache (declared in models.txt),
formats each (subject, item, benchmark, condition) tuple as a yes/no
completion prompt, reads the next-token log-probabilities for "yes" and
"no", and renormalises to produce a calibrated probability.

Module-level code runs once when the container starts. predict() is called
once per hidden (model_id, item_id) pair.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants / smoke-test helpers (reused from hf_submission template)
# ---------------------------------------------------------------------------

LOCAL_SMOKE_TEST_ENV = "PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"


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


def _single_declared_model() -> str:
    declared = _declared_models()
    if not declared:
        raise RuntimeError(
            "models.txt is missing or empty. Declare exactly one HuggingFace repo."
        )
    if len(declared) > 1:
        raise RuntimeError(
            f"models.txt declares {len(declared)} repos; expected exactly one."
        )
    return declared[0]


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
# Judge prompt
# ---------------------------------------------------------------------------

# Values < 1 sharpen predictions toward 0/1; values > 1 flatten toward 0.5.
SHARPENING_TEMPERATURE = 0.75

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
# Module-level init
# ---------------------------------------------------------------------------

REPO_ID = _single_declared_model()

TOKENIZER = None
MODEL = None
YES_ID: int | None = None
NO_ID: int | None = None

# Cache zero-shot predictions so acquisition_function() and predict() share work.
_CACHE: dict[str, float] = {}

if _local_smoke_test_enabled():
    print(f"[model] Skipped HuggingFace load for local smoke test.", flush=True)
else:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "torch and transformers are required. "
            "Use sample_code_submission/ for a no-HF baseline."
        ) from exc

    try:
        _cache_dir = _resolve_cache_dir()
        TOKENIZER = AutoTokenizer.from_pretrained(
            REPO_ID,
            cache_dir=_cache_dir,
            local_files_only=True,
        )
        MODEL = AutoModelForCausalLM.from_pretrained(
            REPO_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=_cache_dir,
            local_files_only=True,
        )
        MODEL.eval()
        YES_ID = TOKENIZER.encode("yes", add_special_tokens=False)[-1]
        NO_ID = TOKENIZER.encode("no", add_special_tokens=False)[-1]
        print(f"[model] Loaded {REPO_ID}.", flush=True)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load '{REPO_ID}' from the local HF cache. "
            "Check models.txt and the platform pre-download step."
        ) from exc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cache_key(input: dict) -> str:
    return json.dumps(
        {k: input[k] for k in ("benchmark", "condition", "subject_content", "item_content")},
        sort_keys=True,
    )


def _build_prompt(input: dict, labeled: list[dict] | None) -> str:
    """Construct a raw-completion prompt, optionally with few-shot examples."""
    prefix = ""
    if labeled:
        same_bench = [ex for ex in labeled if ex.get("benchmark") == input.get("benchmark")]
        examples = same_bench[:4] if same_bench else labeled[:4]
        for ex in examples:
            ans = "yes" if ex["label"] == 1 else "no"
            prefix += JUDGE_TEMPLATE.format(**ex) + f" {ans}\n\n"
    return prefix + JUDGE_TEMPLATE.format(**input)


def _infer(input: dict, labeled: list[dict] | None) -> float:
    """Run one forward pass and return P(yes | prompt)."""
    import torch

    prompt = _build_prompt(input, labeled)
    ids = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
    with torch.no_grad():
        logits = MODEL(**ids).logits[0, -1]
    lp = torch.log_softmax(logits, dim=-1)
    log_odds = (lp[YES_ID] - lp[NO_ID]).item()
    return float(1.0 / (1.0 + math.exp(-log_odds / SHARPENING_TEMPERATURE)))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def predict(input: dict, labeled: list[dict] | None = None) -> float:
    """Return P(subject answers item correctly)."""
    if MODEL is None:
        return 0.5

    key = _cache_key(input)
    if not labeled and key in _CACHE:
        return _CACHE[key]

    result = _infer(input, labeled)

    if not labeled:
        _CACHE[key] = result
    return result
