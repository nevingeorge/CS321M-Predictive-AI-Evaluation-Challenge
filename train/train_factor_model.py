"""Stage 1: Fit a K-factor logistic model on the public training response matrix.

P(Y_ij = 1 | U_i, V_j, Z_j) = sigmoid(U_i^T V_j + Z_j)

Saves to train/artifacts/:
  U.npy              [n_subjects, K] subject ability matrix
  V.npy              [n_items, K]    item loading matrix
  Z.npy              [n_items, 1]    item difficulty intercepts
  subject_index.json {subject_id: row_idx}
  item_index.json    {item_id: row_idx}
  item_metadata.json {item_id: {benchmark, condition, item_content}}
  subject_metadata.json {subject_id: {display_name, params, release_date, family, organization}}

Run from the starting_kit root:
  python train/train_factor_model.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

K = 4
LAMBDA_REG = 1e-4   # ℓ2 regularization on U and V
LR = 1e-2
EPOCHS = 50
BATCH_SIZE = 4096
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REPO_ID = "aims-foundations/measurement-db"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data loading (per README: explicit parquet files, no traces, no auto-split)
# ---------------------------------------------------------------------------

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
    print(f"Found {len(response_files)} response parquet files.")

    response_features = Features({
        "subject_id": Value("string"),
        "item_id": Value("string"),
        "benchmark_id": Value("string"),
        "trial": Value("int64"),
        "test_condition": Value("string"),
        "response": Value("float64"),
        "correct_answer": Value("string"),
        "trace": Value("string"),
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

    subject_ids, item_ids, labels = [], [], []
    item_metadata, subject_metadata = {}, {}

    # Deduplicate on (subject_id, item_id, test_condition), keep smallest trial
    seen: dict[tuple, int] = {}   # key → label
    for row in responses:
        y = row["response"]
        if y not in (0.0, 1.0):
            continue
        key = (row["subject_id"], row["item_id"], row["test_condition"] or "none")
        if key not in seen:
            seen[key] = int(y)

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

    print(f"Loaded {len(labels):,} binary triples, "
          f"{len(set(subject_ids))} subjects, {len(set(item_ids))} items.")
    return subject_ids, item_ids, labels, item_metadata, subject_metadata


# ---------------------------------------------------------------------------
# Factor model
# ---------------------------------------------------------------------------

class FactorModel(nn.Module):
    def __init__(self, n_subjects: int, n_items: int, k: int):
        super().__init__()
        self.U = nn.Embedding(n_subjects, k)
        self.V = nn.Embedding(n_items, k)
        self.Z = nn.Embedding(n_items, 1)
        nn.init.normal_(self.U.weight, std=0.1)
        nn.init.normal_(self.V.weight, std=0.1)
        nn.init.zeros_(self.Z.weight)

    def forward(self, subj_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u = self.U(subj_idx)            # [B, K]
        v = self.V(item_idx)            # [B, K]
        z = self.Z(item_idx).squeeze(1) # [B]
        logit = (u * v).sum(dim=1) + z  # [B]
        return logit


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model: FactorModel, subj_idx: torch.Tensor, item_idx: torch.Tensor,
          labels: torch.Tensor) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    dataset = TensorDataset(subj_idx, item_idx, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for si, ii, y in loader:
            si, ii, y = si.to(DEVICE), ii.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logit = model(si, ii)
            loss = bce(logit, y)
            # ℓ2 regularization on U and V
            loss = loss + LAMBDA_REG * (
                model.U.weight.norm(2) ** 2 + model.V.weight.norm(2) ** 2
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS}  loss={total_loss:.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    responses, items, subjects = load_training_data()
    subject_ids, item_ids, labels, item_metadata, subject_metadata = build_triples(
        responses, items, subjects
    )

    # Build index maps
    all_subjects = sorted(set(subject_ids))
    all_items = sorted(set(item_ids))
    subject_index = {sid: i for i, sid in enumerate(all_subjects)}
    item_index = {iid: i for i, iid in enumerate(all_items)}

    si_tensor = torch.tensor([subject_index[s] for s in subject_ids], dtype=torch.long)
    ii_tensor = torch.tensor([item_index[i] for i in item_ids], dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.float32)

    model = FactorModel(len(all_subjects), len(all_items), K).to(DEVICE)
    print(f"Training factor model: {len(all_subjects)} subjects, "
          f"{len(all_items)} items, K={K}, device={DEVICE}")
    train(model, si_tensor, ii_tensor, y_tensor)

    # Save artifacts
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
    print(f"Artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
