"""Stage 3 (v5): Tune ensemble weights on the validation split.

Loads v5_val_triples.json (saved by train_factor_model_v5.py), runs both the
factor model and the LLM judge on each triple, then minimises binary
cross-entropy over [w_factor, w_llm, bias] using L-BFGS-B.

Saves v5/ensemble_weights.json to HF.

Run via Modal after Stages 1 and 2:
  modal run train/train_modal_v5.py --stage tune
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
HF_REPO = "ngeorge/cs321m-competition"
MAX_TUNE_SAMPLES = 10_000   # cap LLM inference; 537K val triples would take ~30h
ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
ENCODER_DIM = 1024
K = 4

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
# Subject content reconstruction
# ---------------------------------------------------------------------------

def _render_subject(display_name: str, meta: dict) -> str:
    lines = [f"Name: {display_name}"]
    for key, label in [("organization", "Organization"), ("params", "Parameters"),
                        ("release_date", "Released"), ("family", "Family")]:
        v = meta.get(key, "")
        if v:
            lines.append(f"{label}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factor model logit
# ---------------------------------------------------------------------------

def compute_factor_logits(val_triples, item_metadata, subject_metadata,
                          subject_index, U, encoder, item_head) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    subject_metadata_by_display = {
        (m.get("display_name") or sid): (sid, m)
        for sid, m in subject_metadata.items()
    }

    logits = []
    for triple in val_triples:
        sid = triple["subject_id"]
        iid = triple["item_id"]
        item_meta = item_metadata.get(iid, {})

        item_text = (
            f"Benchmark: {item_meta.get('benchmark', '')}\n"
            f"Condition: {item_meta.get('condition', 'none')}\n"
            f"Item: {item_meta.get('item_content', '')}"
        )
        emb = torch.tensor(encoder.encode(item_text, normalize_embeddings=True),
                           dtype=torch.float32)
        with torch.no_grad():
            out = item_head(emb)
        v_hat, z_hat = out[:K], out[K]

        if sid in subject_index:
            u = torch.tensor(U[subject_index[sid]], dtype=torch.float32)
        else:
            u = torch.zeros(K)

        logit = float(((u * v_hat).sum() + z_hat).item())
        logits.append(logit)
    return np.array(logits)


# ---------------------------------------------------------------------------
# LLM logit
# ---------------------------------------------------------------------------

def compute_llm_logits(val_triples, item_metadata, subject_metadata, llm, tokenizer,
                       yes_id, no_id) -> np.ndarray:
    logits = []
    for i, triple in enumerate(val_triples):
        if i % 100 == 0:
            print(f"  LLM logit {i}/{len(val_triples)}...")
        sid = triple["subject_id"]
        iid = triple["item_id"]
        item_meta = item_metadata.get(iid, {})
        subj_meta = subject_metadata.get(sid, {})
        subject_content = _render_subject(subj_meta.get("display_name", sid), subj_meta)

        inp = {
            "benchmark": item_meta.get("benchmark", ""),
            "condition": item_meta.get("condition", "none"),
            "subject_content": subject_content,
            "item_content": item_meta.get("item_content", ""),
        }
        prompt = JUDGE_TEMPLATE.format(**inp)
        ids = tokenizer(prompt, return_tensors="pt").to(llm.device)
        with torch.no_grad():
            logit_vec = llm(**ids).logits[0, -1]
        lp = torch.log_softmax(logit_vec, dim=-1)
        logits.append(float(lp[yes_id].item() - lp[no_id].item()))
    return np.array(logits)


# ---------------------------------------------------------------------------
# Optimise [w_factor, w_llm, bias]
# ---------------------------------------------------------------------------

def optimise_weights(logits_factor, logits_llm, labels):
    from scipy.optimize import minimize

    def loss(params):
        wf, wl, b = params
        logit = wf * logits_factor + wl * logits_llm + b
        p = 1.0 / (1.0 + np.exp(-logit))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))

    result = minimize(loss, [0.5, 0.5, 0.0], method="L-BFGS-B",
                      bounds=[(-10, 10), (-10, 10), (-10, 10)])
    wf, wl, b = result.x
    print(f"Optimised weights: w_factor={wf:.4f}  w_llm={wl:.4f}  bias={b:.4f}")
    print(f"Val log-loss: {result.fun:.6f}")
    return {"w_factor": float(wf), "w_llm": float(wl), "bias": float(b)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading artifacts...")
    U = np.load(ARTIFACTS_DIR / "U.npy")
    subject_index = json.loads((ARTIFACTS_DIR / "subject_index.json").read_text())
    item_metadata = json.loads((ARTIFACTS_DIR / "item_metadata.json").read_text())
    subject_metadata = json.loads((ARTIFACTS_DIR / "subject_metadata.json").read_text())
    import random
    val_triples = json.loads((ARTIFACTS_DIR / "v5_val_triples.json").read_text())
    if len(val_triples) > MAX_TUNE_SAMPLES:
        random.seed(42)
        val_triples = random.sample(val_triples, MAX_TUNE_SAMPLES)
        print(f"Sampled {MAX_TUNE_SAMPLES:,} triples from val set for ensemble tuning.")
    nn_weights = torch.load(ARTIFACTS_DIR / "v5_nn_weights.pt", map_location="cpu")

    import torch.nn as nn
    item_head = nn.Sequential(nn.Linear(ENCODER_DIM, 256), nn.ReLU(), nn.Linear(256, K + 1))
    item_head.load_state_dict(nn_weights["item_head"])
    item_head.eval()

    labels = np.array([t["label"] for t in val_triples], dtype=np.float64)

    # Factor model logits
    print("\n=== Computing factor model logits ===")
    encoder = SentenceTransformer(ENCODER_MODEL)
    logits_factor = compute_factor_logits(
        val_triples, item_metadata, subject_metadata, subject_index, U, encoder, item_head
    )

    # LLM logits
    print("\n=== Computing LLM logits ===")
    llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3.5-9B", torch_dtype=torch.bfloat16, device_map="auto"
    )
    llm.eval()
    yes_id = llm_tokenizer.encode("yes", add_special_tokens=False)[-1]
    no_id = llm_tokenizer.encode("no", add_special_tokens=False)[-1]
    logits_llm = compute_llm_logits(
        val_triples, item_metadata, subject_metadata, llm, llm_tokenizer, yes_id, no_id
    )

    # Optimise
    print("\n=== Optimising ensemble weights ===")
    weights = optimise_weights(logits_factor, logits_llm, labels)

    # Save and upload
    weights_json = json.dumps(weights, indent=2)
    (ARTIFACTS_DIR / "v5_ensemble_weights.json").write_text(weights_json)

    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=weights_json.encode(),
            path_in_repo="v5/ensemble_weights.json",
            repo_id=HF_REPO, repo_type="model",
        )
        print(f"Uploaded v5/ensemble_weights.json → {HF_REPO}")
    else:
        print("HF_TOKEN not set — skipped upload.")

    print(f"\nEnsemble weights: {weights}")


if __name__ == "__main__":
    main()
