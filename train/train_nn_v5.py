"""Stage 2 (v5): Item MLP + model linear layer with improvements.

Changes from v4:
  - Encoder: BAAI/bge-large-en-v1.5 (1024-dim instead of 768-dim)
  - Stage 2b: adds is_instruct binary feature
  - Joint fine-tuning: 3 epochs of end-to-end training with small LR
  - Artifacts saved under v5/ prefix in the HF repo

Run from starting_kit root after train_factor_model_v5.py:
  python train/train_nn_v5.py
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

K = 4
ENCODER_DIM = 1024          # BAAI/bge-large-en-v1.5
ENCODER_MODEL = "BAAI/bge-large-en-v1.5"
LR_ITEM = 1e-3
LR_MODEL = 1e-3
LR_JOINT = 1e-5             # for encoder layers during joint fine-tuning
EPOCHS_ITEM = 30
EPOCHS_MODEL = 100
EPOCHS_JOINT = 3
BATCH_SIZE = 2048
JOINT_BATCH_SIZE = 16       # small batch for encoder backprop; avoids OOM
JOINT_MAX_SAMPLES = 100_000  # cap triples used for joint fine-tuning
TOP_N_FAMILIES = 20
TOP_N_ORGS = 20
REFERENCE_DATE = datetime(2020, 1, 1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
SUBMISSION_DIR = Path(__file__).parents[1] / "my_submission_v5_ensemble"
HF_REPO = "ngeorge/cs321m-competition"


# ---------------------------------------------------------------------------
# Feature helpers (Stage 2b)
# ---------------------------------------------------------------------------

def _parse_params_log(s: str) -> float:
    if not s:
        return 0.0
    s = s.strip().upper()
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


def _is_instruct(meta: dict) -> float:
    text = (meta.get("display_name", "") + " " + meta.get("family", "")).lower()
    return 1.0 if any(k in text for k in ["instruct", "chat", "-it", "rlhf"]) else 0.0


def _build_vocab(items: list[str], top_n: int) -> dict[str, int]:
    from collections import Counter
    vocab = {"<other>": 0}
    for name, _ in Counter(items).most_common(top_n):
        if name:
            vocab[name] = len(vocab)
    return vocab


def _onehot(value: str, vocab: dict[str, int]) -> list[float]:
    idx = vocab.get(value, vocab["<other>"])
    vec = [0.0] * len(vocab)
    vec[idx] = 1.0
    return vec


def build_feature_config(subject_metadata: dict) -> dict:
    families = [m.get("family", "") for m in subject_metadata.values()]
    orgs = [m.get("organization", "") for m in subject_metadata.values()]
    family_vocab = _build_vocab(families, TOP_N_FAMILIES)
    org_vocab = _build_vocab(orgs, TOP_N_ORGS)
    d_F = 3 + len(family_vocab) + len(org_vocab)   # log_params + release_days + is_instruct + one-hots
    return {"family_vocab": family_vocab, "org_vocab": org_vocab, "d_F": d_F}


def subject_to_features(meta: dict, config: dict) -> list[float]:
    return (
        [_parse_params_log(meta.get("params", "")),
         _parse_release_days(meta.get("release_date", "")),
         _is_instruct(meta)]
        + _onehot(meta.get("family", ""), config["family_vocab"])
        + _onehot(meta.get("organization", ""), config["org_vocab"])
    )


# ---------------------------------------------------------------------------
# Stage 2a: Item MLP  (Linear(1024,256) → ReLU → Linear(256, K+1))
# ---------------------------------------------------------------------------

def build_item_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(ENCODER_DIM, 256), nn.ReLU(), nn.Linear(256, K + 1)
    )


def encode_items(item_metadata: dict, encoder) -> dict[str, np.ndarray]:
    item_ids = list(item_metadata.keys())
    texts = [
        f"Benchmark: {item_metadata[iid]['benchmark']}\n"
        f"Condition: {item_metadata[iid]['condition']}\n"
        f"Item: {item_metadata[iid]['item_content']}"
        for iid in item_ids
    ]
    print(f"Encoding {len(texts)} items with {ENCODER_MODEL}...")
    embs = encoder.encode(texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True)
    return {iid: embs[i] for i, iid in enumerate(item_ids)}


def train_item_head(item_head, item_embs, U, subject_ids, item_ids_list, labels,
                    subject_index, item_index):
    U_tensor = torch.tensor(U, dtype=torch.float32).to(DEVICE)
    si_list, emb_list, y_list = [], [], []
    for sid, iid, y in zip(subject_ids, item_ids_list, labels):
        if sid in subject_index and iid in item_embs:
            si_list.append(subject_index[sid])
            emb_list.append(item_embs[iid])
            y_list.append(float(y))

    si_t = torch.tensor(si_list, dtype=torch.long)
    emb_t = torch.tensor(np.array(emb_list), dtype=torch.float32)
    y_t = torch.tensor(y_list, dtype=torch.float32)
    dataset = TensorDataset(si_t, emb_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(item_head.parameters(), lr=LR_ITEM)
    bce = nn.BCEWithLogitsLoss()

    item_head.to(DEVICE)
    for epoch in range(1, EPOCHS_ITEM + 1):
        item_head.train()
        total = 0.0
        for si, emb, y in loader:
            si, emb, y = si.to(DEVICE), emb.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = item_head(emb)
            v_hat, z_hat = out[:, :K], out[:, K]
            logit = (U_tensor[si] * v_hat).sum(dim=1) + z_hat
            loss = bce(logit, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Item head epoch {epoch:3d}/{EPOCHS_ITEM}  loss={total:.4f}")


# ---------------------------------------------------------------------------
# Joint fine-tuning: unfreeze last 2 encoder layers + item_head end-to-end
# ---------------------------------------------------------------------------

def joint_finetune(item_head, encoder, subject_ids, item_ids_list, labels,
                   item_metadata, subject_index, U):
    """Fine-tune last 2 encoder layers + item_head jointly via masked BCE."""
    print(f"  Joint fine-tuning {EPOCHS_JOINT} epochs (LR encoder={LR_JOINT}, head={LR_ITEM})...")

    raw_model = encoder._first_module().auto_model
    tokenizer = encoder.tokenizer

    # Freeze all encoder params, then unfreeze last 2 transformer layers
    for param in raw_model.parameters():
        param.requires_grad = False
    n_layers = raw_model.config.num_hidden_layers
    for name, param in raw_model.named_parameters():
        if any(f"layer.{n_layers - 1 - i}." in name for i in range(2)):
            param.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": [p for p in raw_model.parameters() if p.requires_grad], "lr": LR_JOINT},
        {"params": item_head.parameters(), "lr": LR_ITEM},
    ])
    bce = nn.BCEWithLogitsLoss()
    U_tensor = torch.tensor(U, dtype=torch.float32).to(DEVICE)
    torch.cuda.empty_cache()
    raw_model.gradient_checkpointing_enable()
    raw_model.to(DEVICE)
    item_head.to(DEVICE)

    # Build text list and indices (capped to avoid multi-hour fine-tuning)
    import random as _random
    triple_pool = list(zip(subject_ids, item_ids_list, labels))
    if len(triple_pool) > JOINT_MAX_SAMPLES:
        _random.seed(0)
        triple_pool = _random.sample(triple_pool, JOINT_MAX_SAMPLES)
    print(f"  Joint fine-tuning on {len(triple_pool):,} triples.")

    texts_all, si_all, y_all = [], [], []
    for sid, iid, y in triple_pool:
        if sid not in subject_index or iid not in item_metadata:
            continue
        meta = item_metadata[iid]
        texts_all.append(
            f"Benchmark: {meta['benchmark']}\nCondition: {meta['condition']}\nItem: {meta['item_content']}"
        )
        si_all.append(subject_index[sid])
        y_all.append(float(y))

    for epoch in range(1, EPOCHS_JOINT + 1):
        raw_model.train()
        item_head.train()
        total = 0.0
        perm = torch.randperm(len(texts_all))
        for start in range(0, len(texts_all), JOINT_BATCH_SIZE):
            idx = perm[start:start + JOINT_BATCH_SIZE].tolist()
            batch_texts = [texts_all[i] for i in idx]
            batch_si = torch.tensor([si_all[i] for i in idx], dtype=torch.long).to(DEVICE)
            batch_y = torch.tensor([y_all[i] for i in idx], dtype=torch.float32).to(DEVICE)

            enc_inputs = tokenizer(batch_texts, padding=True, truncation=True,
                                   max_length=256, return_tensors="pt").to(DEVICE)
            out = raw_model(**enc_inputs)
            # Mean-pool and L2-normalize
            mask = enc_inputs["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)

            head_out = item_head(emb)
            v_hat, z_hat = head_out[:, :K], head_out[:, K]
            logit = (U_tensor[batch_si] * v_hat).sum(dim=1) + z_hat
            loss = bce(logit, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        print(f"  Joint epoch {epoch}/{EPOCHS_JOINT}  loss={total:.4f}")

    # Refreeze encoder
    for param in raw_model.parameters():
        param.requires_grad = False


# ---------------------------------------------------------------------------
# Stage 2b: Model linear layer
# ---------------------------------------------------------------------------

def train_model_head(model_head, F, U):
    F_t = torch.tensor(F, dtype=torch.float32).to(DEVICE)
    U_t = torch.tensor(U, dtype=torch.float32).to(DEVICE)
    model_head.to(DEVICE)
    optimizer = torch.optim.Adam(model_head.parameters(), lr=LR_MODEL)
    for epoch in range(1, EPOCHS_MODEL + 1):
        model_head.train()
        optimizer.zero_grad()
        loss = (model_head(F_t) - U_t).abs().mean()
        loss.backward()
        optimizer.step()
        if epoch % 25 == 0 or epoch == 1:
            print(f"  Model head epoch {epoch:3d}/{EPOCHS_MODEL}  L1={loss.item():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from sentence_transformers import SentenceTransformer

    print("Loading Stage 1 artifacts...")
    U = np.load(ARTIFACTS_DIR / "U.npy")
    subject_index = json.loads((ARTIFACTS_DIR / "subject_index.json").read_text())
    item_index = json.loads((ARTIFACTS_DIR / "item_index.json").read_text())
    item_metadata = json.loads((ARTIFACTS_DIR / "item_metadata.json").read_text())
    subject_metadata = json.loads((ARTIFACTS_DIR / "subject_metadata.json").read_text())

    # Re-load triples for training item head
    print("Re-loading response triples...")
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi
    registry = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    repo_files = HfApi().list_repo_files(repo_id="aims-foundations/measurement-db", repo_type="dataset")
    response_files = sorted(
        n for n in repo_files
        if n.endswith(".parquet") and n not in registry and not n.endswith("_traces.parquet")
    )
    response_features = Features({
        "subject_id": Value("string"), "item_id": Value("string"),
        "benchmark_id": Value("string"), "trial": Value("int64"),
        "test_condition": Value("string"), "response": Value("float64"),
        "correct_answer": Value("string"), "trace": Value("string"),
    })
    responses = load_dataset("aims-foundations/measurement-db",
                             data_files=response_files, features=response_features, split="train")
    seen: dict[tuple, int] = {}
    for row in responses:
        y = row["response"]
        if y not in (0.0, 1.0):
            continue
        key = (row["subject_id"], row["item_id"], row["test_condition"] or "none")
        if key not in seen:
            seen[key] = int(y)
    subject_ids_list = [k[0] for k in seen]
    item_ids_list = [k[1] for k in seen]
    labels = list(seen.values())

    # Stage 2a
    print(f"\n=== Stage 2a: Item head ({ENCODER_MODEL}) ===")
    encoder = SentenceTransformer(ENCODER_MODEL)
    item_embs = encode_items(item_metadata, encoder)
    item_head = build_item_head()
    train_item_head(item_head, item_embs, U, subject_ids_list, item_ids_list,
                    labels, subject_index, item_index)

    # Joint fine-tuning
    print("\n=== Joint fine-tuning ===")
    joint_finetune(item_head, encoder, subject_ids_list, item_ids_list, labels,
                   item_metadata, subject_index, U)
    item_head.cpu().eval()

    # Stage 2b
    print("\n=== Stage 2b: Model head ===")
    feature_config = build_feature_config(subject_metadata)
    d_F = feature_config["d_F"]
    print(f"Feature dim d_F={d_F} (includes is_instruct)")
    ordered = sorted(subject_index, key=lambda s: subject_index[s])
    F = np.array([subject_to_features(subject_metadata.get(s, {}), feature_config)
                  for s in ordered], dtype=np.float32)
    model_head = nn.Linear(d_F, K, bias=True)
    train_model_head(model_head, F, U)
    model_head.cpu().eval()

    # Save
    print("\nSaving...")
    weights = {
        "item_head": item_head.state_dict(),
        "model_head": model_head.state_dict(),
        "feature_config": feature_config,
    }
    torch.save(weights, ARTIFACTS_DIR / "v5_nn_weights.pt")

    # Upload to HF under v5/ prefix
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if token:
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=(ARTIFACTS_DIR / "v5_nn_weights.pt").read_bytes(),
            path_in_repo="v5/nn_weights.pt", repo_id=HF_REPO, repo_type="model",
        )
        # Build and upload subject_abilities.npy (= U.npy) and subject_index.json mapped by display_name
        import io
        subj_abilities = np.load(ARTIFACTS_DIR / "U.npy")
        buf = io.BytesIO()
        np.save(buf, subj_abilities)
        api.upload_file(
            path_or_fileobj=buf.getvalue(),
            path_in_repo="v5/subject_abilities.npy", repo_id=HF_REPO, repo_type="model",
        )
        display_name_index = {
            subject_metadata.get(sid, {}).get("display_name") or sid: idx
            for sid, idx in subject_index.items()
        }
        api.upload_file(
            path_or_fileobj=json.dumps(display_name_index).encode(),
            path_in_repo="v5/subject_index.json", repo_id=HF_REPO, repo_type="model",
        )
        print(f"Uploaded v5/nn_weights.pt, v5/subject_abilities.npy, v5/subject_index.json → {HF_REPO}")
    else:
        np.save(ARTIFACTS_DIR / "v5_subject_abilities.npy", U)
        print("HF_TOKEN not set — skipped upload. Set token to upload.")


if __name__ == "__main__":
    main()
