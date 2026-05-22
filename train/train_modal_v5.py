"""Modal orchestrator for v5 ensemble training pipeline.

Stages:
  1  train_factor_model_v5.py  — K=4 factor model + 15% val split
  2  train_nn_v5.py            — item MLP (bge-large) + model head + joint fine-tuning
  tune  tune_ensemble_v5.py   — optimise w_factor, w_llm, bias on val split

Setup (one-time):
  modal secret create huggingface HF_TOKEN=hf_your_token_here

Usage:
  modal run train/train_modal_v5.py                # all stages
  modal run train/train_modal_v5.py --stage 1
  modal run train/train_modal_v5.py --stage 2
  modal run train/train_modal_v5.py --stage tune
"""

from __future__ import annotations

from pathlib import Path

import modal

TRAIN_DIR = Path(__file__).parent
HF_REPO = "ngeorge/cs321m-competition"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "datasets",
        "huggingface_hub",
        "sentence-transformers",
        "transformers",
        "accelerate",
        "numpy",
        "scipy",
    )
    .add_local_dir(TRAIN_DIR, "/train", copy=True)
    .run_commands("mkdir -p /train/artifacts")
)

app = modal.App("pge-training-v5", image=image)
hf_secret = modal.Secret.from_name("huggingface")
volume = modal.Volume.from_name("pge-artifacts-v5", create_if_missing=True)
REMOTE_ARTIFACTS = Path("/vol/artifacts")


# ---------------------------------------------------------------------------
# Single combined function — runs all stages sequentially in one container.
# Using .spawn() from the local entrypoint means training survives laptop sleep/close.
# ---------------------------------------------------------------------------

@app.function(
    gpu="H100",
    timeout=20 * 3600,          # 20-hour budget covers all three stages
    volumes={"/vol": volume},
    secrets=[hf_secret],
)
def run_all_stages():
    import sys
    sys.path.insert(0, "/train")
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=== Stage 1: factor model ===")
    import train_factor_model_v5 as m1
    m1.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m1.main()
    volume.commit()

    print("=== Stage 2: neural network heads + joint fine-tuning ===")
    import train_nn_v5 as m2
    m2.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m2.SUBMISSION_DIR = REMOTE_ARTIFACTS / "submission"
    m2.main()
    volume.commit()

    print("=== Stage 3: ensemble weight tuning ===")
    import tune_ensemble_v5 as m3
    m3.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m3.main()
    volume.commit()

    print("All stages complete!")


@app.function(
    gpu="H100",
    timeout=16 * 3600,
    volumes={"/vol": volume},
    secrets=[hf_secret],
)
def run_stages_2_and_3():
    """Run Stage 2 + 3 only, using Stage 1 artifacts already in the volume."""
    import sys
    sys.path.insert(0, "/train")
    REMOTE_ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("=== Stage 2: neural network heads + joint fine-tuning ===")
    import train_nn_v5 as m2
    m2.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m2.SUBMISSION_DIR = REMOTE_ARTIFACTS / "submission"
    m2.main()
    volume.commit()

    print("=== Stage 3: ensemble weight tuning ===")
    import tune_ensemble_v5 as m3
    m3.ARTIFACTS_DIR = REMOTE_ARTIFACTS
    m3.main()
    volume.commit()

    print("Stages 2 + 3 complete!")


# ---------------------------------------------------------------------------
# Local entrypoint
# Usage:
#   modal run train/train_modal_v5.py              # all stages
#   modal run train/train_modal_v5.py --from 2     # skip Stage 1, use saved artifacts
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(from_stage: int = 1):
    if from_stage <= 1:
        print("Running all v5 stages on Modal H100. Keep laptop awake until done.")
        run_all_stages.remote()
    else:
        print("Running Stage 2 + 3 using saved Stage 1 artifacts. Keep laptop awake.")
        run_stages_2_and_3.remote()
    print("\nDone. Package the submission next.")
