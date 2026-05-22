"""Modal wrapper for PGE two-stage training.

Runs Stage 1 (factor model) and Stage 2 (neural network heads) on a GPU,
persists artifacts in a Modal Volume, then uploads the three submission
artifacts (nn_weights.pt, subject_abilities.npy, subject_index.json) directly
to a HuggingFace model repository.

Setup (one-time):
  1. Create a HuggingFace model repo at huggingface.co (e.g. yourname/pge-factor-artifacts)
  2. Generate a write token at huggingface.co/settings/tokens
  3. modal secret create huggingface HF_TOKEN=hf_your_token_here
  4. Set HF_REPO below to match your repo name
  5. Update ARTIFACTS_REPO in my_submission_v4_factor/model.py to the same value
  6. Update both lines in my_submission_v4_factor/models.txt

Usage:
  modal run train/train_modal.py                 # run both stages + upload to HF
  modal run train/train_modal.py --stage 1       # Stage 1 only
  modal run train/train_modal.py --stage 2       # Stage 2 only (after Stage 1)
  modal run train/train_modal.py --stage upload  # just upload artifacts to HF
"""

from __future__ import annotations

from pathlib import Path

import modal

# Set this to your HuggingFace model repo (create it first at huggingface.co).
# Must match ARTIFACTS_REPO in my_submission_v4_factor/model.py and models.txt.
HF_REPO = "ngeorge/cs321m-competition"

# ---------------------------------------------------------------------------
# Image — bake training scripts in; pre-create /train/artifacts so the
# module-level mkdir() in the scripts doesn't fail on import
# ---------------------------------------------------------------------------

TRAIN_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "datasets",
        "huggingface_hub",
        "sentence-transformers",
        "numpy",
    )
    .add_local_dir(TRAIN_DIR, "/train", copy=True)
    .run_commands("mkdir -p /train/artifacts")
)

hf_secret = modal.Secret.from_name("huggingface")

app = modal.App("pge-training", image=image)

# Persistent volume for artifacts across Stage 1 → Stage 2
volume = modal.Volume.from_name("pge-artifacts", create_if_missing=True)
REMOTE_ARTIFACTS = Path("/vol/artifacts")


# ---------------------------------------------------------------------------
# Stage 1: K-factor logistic model
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=4 * 3600,
    volumes={"/vol": volume},
)
def run_stage1():
    import sys
    sys.path.insert(0, "/train")
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    import train_factor_model as m
    m.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m.main()

    volume.commit()
    print("Stage 1 complete — artifacts committed to volume.")


# ---------------------------------------------------------------------------
# Stage 2: Item MLP + model linear layer
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=4 * 3600,
    volumes={"/vol": volume},
)
def run_stage2():
    import sys
    sys.path.insert(0, "/train")
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    import train_nn as m
    m.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m.SUBMISSION_DIR = REMOTE_ARTIFACTS / "submission"
    m.main()

    volume.commit()
    print("Stage 2 complete — artifacts committed to volume.")


# ---------------------------------------------------------------------------
# Upload artifacts to HuggingFace
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/vol": volume},
    secrets=[hf_secret],
)
def upload_to_hf():
    import os
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id=HF_REPO, repo_type="model", exist_ok=True)

    upload_files = [
        REMOTE_ARTIFACTS / "submission" / "nn_weights.pt",
        REMOTE_ARTIFACTS / "submission" / "subject_abilities.npy",
        REMOTE_ARTIFACTS / "submission" / "subject_index.json",
    ]
    for path in upload_files:
        api.upload_file(
            path_or_fileobj=path.read_bytes(),
            path_in_repo=path.name,
            repo_id=HF_REPO,
            repo_type="model",
        )
        print(f"  Uploaded {path.name} ({path.stat().st_size:,} bytes) → {HF_REPO}")

    print(f"\nAll artifacts uploaded to https://huggingface.co/{HF_REPO}")


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all"):
    if stage in ("1", "all"):
        print("=== Stage 1: factor model ===")
        run_stage1.remote()

    if stage in ("2", "all"):
        print("=== Stage 2: neural network heads ===")
        run_stage2.remote()

    if stage in ("upload", "all"):
        print(f"=== Uploading artifacts to {HF_REPO} ===")
        upload_to_hf.remote()
        print("\nNext steps:")
        print(f"  1. Update ARTIFACTS_REPO in my_submission_v4_factor/model.py → '{HF_REPO}'")
        print(f"  2. Update models.txt second line → '{HF_REPO}'")
        print("  3. Zip and upload to Codabench:"
              " (cd my_submission_v4_factor && zip -r ../my_submission_v4_factor.zip ."
              " -x '*.pyc' -x '__pycache__/*' -x '*/__pycache__/*')")
