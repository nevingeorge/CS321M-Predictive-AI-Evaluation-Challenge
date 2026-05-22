"""Compute the per-(subject, benchmark, condition) average accuracy lookup table.

Reads the public HuggingFace training data and produces lookup.json, which
is bundled directly into the submission ZIP. No GPU or model inference needed.

Output structure:
  {
    "subjects": {
      "GPT-4": {
        "mmlupro|zero-shot": 0.82,   # (benchmark, condition) key
        "mmlupro":           0.79,   # benchmark-only fallback
        "__overall__":       0.74    # subject-level fallback
      },
      ...
    },
    "benchmarks": {
      "mmlupro": 0.63,   # benchmark-level fallback for unknown subjects
      ...
    },
    "global_mean": 0.58  # final fallback
  }

Run from starting_kit root:
  pip3 install datasets huggingface_hub
  python train/compute_lookup.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO_ID = "aims-foundations/measurement-db"
OUT_PATH = Path(__file__).parents[1] / "my_submission_v11_lookup" / "lookup.json"


def main():
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    print("Listing repo files...")
    registry = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}
    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        f for f in repo_files
        if f.endswith(".parquet") and f not in registry and not f.endswith("_traces.parquet")
    )
    print(f"Found {len(response_files)} response files.")

    response_features = Features({
        "subject_id": Value("string"), "item_id": Value("string"),
        "benchmark_id": Value("string"), "trial": Value("int64"),
        "test_condition": Value("string"), "response": Value("float64"),
        "correct_answer": Value("string"), "trace": Value("string"),
    })
    responses = load_dataset(REPO_ID, data_files=response_files,
                             features=response_features, split="train")
    subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")

    # Map subject_id → display_name
    id_to_name = {row["subject_id"]: (row.get("display_name") or row["subject_id"])
                  for row in subjects}
    print(f"Loaded {len(id_to_name)} subjects.")

    # Use separate dicts to avoid conflating subject names and benchmark IDs.
    # subj_sums[(name, bm, cond)], subj_sums[(name, bm)], subj_sums[(name,)]
    # bm_sums[(bm,)]
    # global_sum[()]
    subj_sums:   dict[tuple, float] = defaultdict(float)
    subj_counts: dict[tuple, int]   = defaultdict(int)
    bm_sums:     dict[str, float]   = defaultdict(float)
    bm_counts:   dict[str, int]     = defaultdict(int)
    global_sum   = 0.0
    global_count = 0

    n_skipped = 0
    for row in responses:
        y = row["response"]
        if y not in (0.0, 1.0):
            n_skipped += 1
            continue
        sid   = row["subject_id"]
        name  = id_to_name.get(sid, sid)
        bm    = row["benchmark_id"] or ""
        cond  = row["test_condition"] or "none"
        label = float(y)

        for key in [(name, bm, cond), (name, bm), (name,)]:
            subj_sums[key]   += label
            subj_counts[key] += 1
        if bm:
            bm_sums[bm]   += label
            bm_counts[bm] += 1
        global_sum   += label
        global_count += 1

    print(f"Processed {global_count:,} binary triples ({n_skipped:,} skipped).")

    def subj_avg(key: tuple) -> float | None:
        c = subj_counts.get(key, 0)
        return subj_sums[key] / c if c > 0 else None

    def bm_avg(bm: str) -> float | None:
        c = bm_counts.get(bm, 0)
        return bm_sums[bm] / c if c > 0 else None

    subject_names = {k[0] for k in subj_counts if len(k) >= 1}
    benchmark_ids = set(bm_counts.keys())

    subject_lookup: dict[str, dict] = {}
    for name in subject_names:
        entry: dict[str, float] = {}
        for key in subj_counts:
            if len(key) == 3 and key[0] == name:
                _, bm, cond = key
                v = subj_avg(key)
                if v is not None:
                    entry[f"{bm}|{cond}"] = round(v, 6)
        for key in subj_counts:
            if len(key) == 2 and key[0] == name:
                _, bm = key
                v = subj_avg(key)
                if v is not None:
                    entry[bm] = round(v, 6)
        overall = subj_avg((name,))
        if overall is not None:
            entry["__overall__"] = round(overall, 6)
        subject_lookup[name] = entry

    benchmark_lookup = {
        bm: round(bm_avg(bm), 6)
        for bm in benchmark_ids
        if bm_avg(bm) is not None
    }

    global_mean = global_sum / global_count if global_count else 0.5
    lookup = {
        "subjects":    subject_lookup,
        "benchmarks":  benchmark_lookup,
        "global_mean": round(global_mean, 6),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(lookup, separators=(",", ":")))
    size = OUT_PATH.stat().st_size
    print(f"Saved lookup.json ({size:,} bytes) → {OUT_PATH}")
    print(f"  {len(subject_lookup)} subjects, {len(benchmark_lookup)} benchmarks")


if __name__ == "__main__":
    main()
