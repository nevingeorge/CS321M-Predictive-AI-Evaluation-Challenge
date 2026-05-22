"""Stage 2: Train item MLP (2a) and model linear layer (2b).

Stage 2a — Item branch:
  sentence_encoder(item_text) -> Linear(768,256) -> ReLU -> Linear(256, K+1)
  Output: [V_hat_j (K-dim), Z_hat_j (scalar)]
  Loss: masked BCE with frozen U from Stage 1

Stage 2b — Model branch:
  structured_features(subject_metadata) -> Linear(d_F, K)
  Features: log_params, release_days, family one-hot, org one-hot
  Loss: ℓ1 regression to Stage 1 U estimates

Run from the starting_kit root AFTER train_factor_model.py:
  python train/train_nn.py

Copies trained weights to my_submission_v4_factor/:
  nn_weights.pt
  subject_abilities.npy  (= U.npy)
  subject_index.json
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K = 4
LR_ITEM = 1e-3
LR_MODEL = 1e-3
EPOCHS_ITEM = 30
EPOCHS_MODEL = 100
BATCH_SIZE = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER_MODEL = "all-mpnet-base-v2"   # 768-dim sentence transformer
TOP_N_FAMILIES = 20
TOP_N_ORGS = 20
REFERENCE_DATE = datetime(2020, 1, 1)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SUBMISSION_DIR = Path(__file__).parents[1] / "my_submission_v4_factor"


# ---------------------------------------------------------------------------
# Feature engineering for subject metadata (Stage 2b)
# ---------------------------------------------------------------------------

def parse_params_log(params_str: str) -> float:
    """Parse '7B' or '70B' or '1.76T' to log10(billions). Returns 0.0 on failure."""
    if not params_str:
        return 0.0
    s = params_str.strip().upper()
    try:
        if s.endswith("T"):
            return float(np.log10(float(s[:-1]) * 1000))
        if s.endswith("B"):
            return float(np.log10(max(float(s[:-1]), 1e-3)))
        if s.endswith("M"):
            return float(np.log10(max(float(s[:-1]) / 1000, 1e-6)))
        return float(np.log10(max(float(s), 1e-3)))
    except ValueError:
        return 0.0


def parse_release_days(date_str: str) -> float:
    """Parse 'YYYY-MM-DD' to days since 2020-01-01. Returns 0.0 on failure."""
    if not date_str:
        return 0.0
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            d = datetime.strptime(date_str.strip(), fmt)
            return float((d - REFERENCE_DATE).days)
        except ValueError:
            continue
    return 0.0


def build_vocab(items: list[str], top_n: int) -> dict[str, int]:
    from collections import Counter
    counts = Counter(items)
    vocab = {"<other>": 0}
    for name, _ in counts.most_common(top_n):
        if name:
            vocab[name] = len(vocab)
    return vocab


def make_onehot(value: str, vocab: dict[str, int]) -> list[float]:
    idx = vocab.get(value, vocab["<other>"])
    vec = [0.0] * len(vocab)
    vec[idx] = 1.0
    return vec


def build_feature_config(subject_metadata: dict) -> dict:
    families = [m.get("family", "") for m in subject_metadata.values()]
    orgs = [m.get("organization", "") for m in subject_metadata.values()]
    family_vocab = build_vocab(families, TOP_N_FAMILIES)
    org_vocab = build_vocab(orgs, TOP_N_ORGS)
    d_F = 2 + len(family_vocab) + len(org_vocab)   # log_params + release_days + one-hots
    return {"family_vocab": family_vocab, "org_vocab": org_vocab, "d_F": d_F}


def subject_to_features(meta: dict, config: dict) -> list[float]:
    log_p = parse_params_log(meta.get("params", ""))
    rel_d = parse_release_days(meta.get("release_date", ""))
    fam_oh = make_onehot(meta.get("family", ""), config["family_vocab"])
    org_oh = make_onehot(meta.get("organization", ""), config["org_vocab"])
    return [log_p, rel_d] + fam_oh + org_oh


# ---------------------------------------------------------------------------
# Stage 2a: Item MLP
# ---------------------------------------------------------------------------

def build_item_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Linear(256, K + 1),
    )


def encode_items(item_metadata: dict, encoder) -> tuple[dict[str, np.ndarray], list[str]]:
    """Encode all unique item texts. Returns {item_id: embedding}."""
    from sentence_transformers import SentenceTransformer
    item_ids = list(item_metadata.keys())
    texts = [
        f"Benchmark: {item_metadata[iid]['benchmark']}\n"
        f"Condition: {item_metadata[iid]['condition']}\n"
        f"Item: {item_metadata[iid]['item_content']}"
        for iid in item_ids
    ]
    print(f"Encoding {len(texts)} item texts...")
    embs = encoder.encode(texts, batch_size=256, show_progress_bar=True)
    return {iid: embs[i] for i, iid in enumerate(item_ids)}, item_ids


def train_item_head(item_head: nn.Module, item_embs: dict[str, np.ndarray],
                    U: np.ndarray, subject_ids: list[str], item_ids_list: list[str],
                    labels: list[int], subject_index: dict, item_index: dict) -> None:
    """Masked BCE with frozen U."""
    U_tensor = torch.tensor(U, dtype=torch.float32).to(DEVICE)

    # Build (subject_row, item_emb, label) for observed entries
    si_list, emb_list, y_list = [], [], []
    for sid, iid, y in zip(subject_ids, item_ids_list, labels):
        if sid in subject_index and iid in item_embs:
            si_list.append(subject_index[sid])
            emb_list.append(item_embs[iid])
            y_list.append(float(y))

    si_tensor = torch.tensor(si_list, dtype=torch.long)
    emb_tensor = torch.tensor(np.array(emb_list), dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32)

    dataset = TensorDataset(si_tensor, emb_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(item_head.parameters(), lr=LR_ITEM)
    bce = nn.BCEWithLogitsLoss()

    item_head.to(DEVICE)
    for epoch in range(1, EPOCHS_ITEM + 1):
        item_head.train()
        total_loss = 0.0
        for si, emb, y in loader:
            si, emb, y = si.to(DEVICE), emb.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = item_head(emb)            # [B, K+1]
            v_hat = out[:, :K]             # [B, K]
            z_hat = out[:, K]              # [B]
            u = U_tensor[si]               # [B, K]  — frozen
            logit = (u * v_hat).sum(dim=1) + z_hat
            loss = bce(logit, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Item head epoch {epoch:3d}/{EPOCHS_ITEM}  loss={total_loss:.4f}")


# ---------------------------------------------------------------------------
# Stage 2b: Model linear layer
# ---------------------------------------------------------------------------

def train_model_head(model_head: nn.Linear, F: np.ndarray, U: np.ndarray) -> None:
    """ℓ1 regression: F @ W ≈ U."""
    F_tensor = torch.tensor(F, dtype=torch.float32).to(DEVICE)
    U_tensor = torch.tensor(U, dtype=torch.float32).to(DEVICE)
    model_head.to(DEVICE)
    optimizer = torch.optim.Adam(model_head.parameters(), lr=LR_MODEL)
    for epoch in range(1, EPOCHS_MODEL + 1):
        model_head.train()
        optimizer.zero_grad()
        u_hat = model_head(F_tensor)
        loss = (u_hat - U_tensor).abs().mean()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Model head epoch {epoch:3d}/{EPOCHS_MODEL}  L1={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading Stage 1 artifacts...")
    U = np.load(ARTIFACTS_DIR / "U.npy")
    subject_index = json.loads((ARTIFACTS_DIR / "subject_index.json").read_text())
    item_index = json.loads((ARTIFACTS_DIR / "item_index.json").read_text())
    item_metadata = json.loads((ARTIFACTS_DIR / "item_metadata.json").read_text())
    subject_metadata = json.loads((ARTIFACTS_DIR / "subject_metadata.json").read_text())

    # Reload original triple lists for Stage 2a training
    # Re-derive from Stage 1 artifacts (item_index covers all observed items)
    # We need (subject_id, item_id, label) — load from HuggingFace again
    # or re-derive from the saved indices. For simplicity, reload.
    print("Re-loading response triples for Stage 2a training...")
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    registry_files = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    repo_files = HfApi().list_repo_files(repo_id="aims-foundations/measurement-db",
                                         repo_type="dataset")
    response_files = sorted(
        name for name in repo_files
        if name.endswith(".parquet")
        and name not in registry_files
        and not name.endswith("_traces.parquet")
    )
    response_features = Features({
        "subject_id": Value("string"), "item_id": Value("string"),
        "benchmark_id": Value("string"), "trial": Value("int64"),
        "test_condition": Value("string"), "response": Value("float64"),
        "correct_answer": Value("string"), "trace": Value("string"),
    })
    responses = load_dataset("aims-foundations/measurement-db",
                             data_files=response_files,
                             features=response_features, split="train")

    seen: dict[tuple, int] = {}
    for row in responses:
        y = row["response"]
        if y not in (0.0, 1.0):
            continue
        key = (row["subject_id"], row["item_id"], row["test_condition"] or "none")
        if key not in seen:
            seen[key] = int(y)

    subject_ids_list = [k[0] for k in seen]
    item_ids_triple = [k[1] for k in seen]
    labels = list(seen.values())

    # ---------- Stage 2a ----------
    print("\n=== Stage 2a: Training item head ===")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER_MODEL)
    item_embs, _ = encode_items(item_metadata, encoder)

    item_head = build_item_head()
    train_item_head(item_head, item_embs, U, subject_ids_list, item_ids_triple,
                    labels, subject_index, item_index)
    item_head.cpu().eval()

    # ---------- Stage 2b ----------
    print("\n=== Stage 2b: Training model head ===")
    feature_config = build_feature_config(subject_metadata)
    d_F = feature_config["d_F"]
    print(f"Feature dimension d_F = {d_F}")

    ordered_subjects = sorted(subject_index, key=lambda s: subject_index[s])
    F_list = []
    for sid in ordered_subjects:
        meta = subject_metadata.get(sid, {})
        F_list.append(subject_to_features(meta, feature_config))
    F = np.array(F_list, dtype=np.float32)

    model_head = nn.Linear(d_F, K, bias=True)
    train_model_head(model_head, F, U)
    model_head.cpu().eval()

    # ---------- Save ----------
    print("\nSaving weights...")
    weights = {
        "item_head": item_head.state_dict(),
        "model_head": model_head.state_dict(),
        "feature_config": feature_config,
    }
    torch.save(weights, ARTIFACTS_DIR / "nn_weights.pt")
    print(f"Saved nn_weights.pt to {ARTIFACTS_DIR}/")

    # Copy to submission directory
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(ARTIFACTS_DIR / "nn_weights.pt", SUBMISSION_DIR / "nn_weights.pt")
    shutil.copy(ARTIFACTS_DIR / "U.npy", SUBMISSION_DIR / "subject_abilities.npy")
    (SUBMISSION_DIR / "subject_index.json").write_text(
        json.dumps({subject_metadata.get(sid, {}).get("display_name") or sid: idx
                    for sid, idx in subject_index.items()})
    )
    print(f"Copied artifacts to {SUBMISSION_DIR}/")


if __name__ == "__main__":
    main()
