"""Stage 1 (v5): K=4 factor model with 15% validation split.

Same as train_factor_model.py but holds out 15% of triples for ensemble tuning.
Saves v5_val_triples.json alongside the standard artifacts.

Run from starting_kit root:
  python train/train_factor_model_v5.py
Or via Modal:
  modal run train/train_modal_v5.py --stage 1
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

K = 4
LAMBDA_REG = 1e-4
LR = 1e-2
EPOCHS = 50
BATCH_SIZE = 4096
VAL_FRACTION = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "aims-foundations/measurement-db"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data():
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    print("Listing HuggingFace repo files...")
    registry_files = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
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
    print("Loading response data...")
    responses = load_dataset(REPO_ID, data_files=response_files,
                             features=response_features, split="train")
    items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
    return responses, items, subjects


def build_triples(responses, items, subjects):
    items_by_id = {row["item_id"]: row for row in items}
    subjects_by_id = {row["subject_id"]: row for row in subjects}
    item_metadata, subject_metadata = {}, {}

    seen: dict[tuple, int] = {}
    for row in responses:
        y = row["response"]
        if y not in (0.0, 1.0):
            continue
        key = (row["subject_id"], row["item_id"], row["test_condition"] or "none")
        if key not in seen:
            seen[key] = int(y)

    subject_ids, item_ids, labels = [], [], []
    for (sid, iid, _), label in seen.items():
        subject_ids.append(sid)
        item_ids.append(iid)
        labels.append(label)
        if iid not in item_metadata and iid in items_by_id:
            row = items_by_id[iid]
            item_metadata[iid] = {
                "benchmark": row.get("benchmark_id", ""),
                "condition": row.get("test_condition", "none") or "none",
                "item_content": row.get("content", ""),
            }
        if sid not in subject_metadata and sid in subjects_by_id:
            row = subjects_by_id[sid]
            subject_metadata[sid] = {
                "display_name": row.get("display_name") or sid,
                "params": row.get("params", ""),
                "release_date": row.get("release_date", ""),
                "family": row.get("family", ""),
                "organization": row.get("provider", ""),
            }

    print(f"Loaded {len(labels):,} binary triples.")
    return subject_ids, item_ids, labels, item_metadata, subject_metadata


class FactorModel(nn.Module):
    def __init__(self, n_subjects, n_items, k):
        super().__init__()
        self.U = nn.Embedding(n_subjects, k)
        self.V = nn.Embedding(n_items, k)
        self.Z = nn.Embedding(n_items, 1)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)
        nn.init.zeros_(self.Z.weight)

    def forward(self, si, ii):
        u = self.U(si)
        v = self.V(ii)
        z = self.Z(ii).squeeze(1)
        return (u * v).sum(dim=1) + z


def train(model, si, ii, labels):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    dataset = TensorDataset(si, ii, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for s, i, y in loader:
            s, i, y = s.to(DEVICE), i.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = bce(model(s, i), y)
            loss = loss + LAMBDA_REG * (model.U.weight.norm(2)**2 + model.V.weight.norm(2)**2)
            loss.backward()
            optimizer.step()
            total += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={total:.1f}")


def main():
    responses, items, subjects = load_training_data()
    subject_ids, item_ids, labels, item_metadata, subject_metadata = build_triples(
        responses, items, subjects
    )

    # Shuffle and split
    indices = list(range(len(labels)))
    random.seed(42)
    random.shuffle(indices)
    n_val = int(len(indices) * VAL_FRACTION)
    val_idx = set(indices[:n_val])
    train_idx = [i for i in indices if i not in val_idx]
    val_idx = sorted(val_idx)

    val_triples = [
        {"subject_id": subject_ids[i], "item_id": item_ids[i], "label": labels[i]}
        for i in val_idx
    ]
    train_sids = [subject_ids[i] for i in train_idx]
    train_iids = [item_ids[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    print(f"Train: {len(train_sids):,}  Val: {len(val_triples):,}")

    all_subjects = sorted(set(subject_ids))
    all_items = sorted(set(item_ids))
    subject_index = {s: i for i, s in enumerate(all_subjects)}
    item_index = {it: i for i, it in enumerate(all_items)}

    si = torch.tensor([subject_index[s] for s in train_sids], dtype=torch.long)
    ii = torch.tensor([item_index[it] for it in train_iids], dtype=torch.long)
    y = torch.tensor(train_labels, dtype=torch.float32)

    model = FactorModel(len(all_subjects), len(all_items), K).to(DEVICE)
    print(f"Training: {len(all_subjects)} subjects, {len(all_items)} items, K={K}")
    train(model, si, ii, y)

    U = model.U.weight.detach().cpu().numpy()
    V = model.V.weight.detach().cpu().numpy()
    Z = model.Z.weight.detach().cpu().numpy()
    np.save(ARTIFACTS_DIR / "U.npy", U)
    np.save(ARTIFACTS_DIR / "V.npy", V)
    np.save(ARTIFACTS_DIR / "Z.npy", Z)
    (ARTIFACTS_DIR / "subject_index.json").write_text(json.dumps(subject_index))
    (ARTIFACTS_DIR / "item_index.json").write_text(json.dumps(item_index))
    (ARTIFACTS_DIR / "item_metadata.json").write_text(json.dumps(item_metadata))
    (ARTIFACTS_DIR / "subject_metadata.json").write_text(json.dumps(subject_metadata))
    (ARTIFACTS_DIR / "v5_val_triples.json").write_text(json.dumps(val_triples))
    print(f"Saved artifacts to {ARTIFACTS_DIR}/  (val triples: {len(val_triples):,})")


if __name__ == "__main__":
    main()
