# CS321M Predictive Evaluation Competition

## Submissions

| Version | Directory | Technical Report | Leaderboard Score |
|---------|-----------|-----------------|-------------------|
| v5 | `my_submission_v5_ensemble/` | Approach 1 | -0.68 |
| v11 | `my_submission_v11_lookup/` | Approach 2 | -0.61 |

### v5 — Approach 1: Ensemble (Factor Model + LLM-as-Judge)

A K=4 factor model combined with a Qwen3.5-9B LLM judge, blended via tuned weights and optionally calibrated with Platt scaling when labeled examples are available.

**Training** (requires [Modal](https://modal.com) and a HuggingFace token):

```bash
# One-time setup
modal secret create huggingface HF_TOKEN=hf_your_token_here

# Run all training stages
modal run train/train_modal_v5.py

# Or run individual stages
modal run train/train_modal_v5.py --stage 1    # factor model
modal run train/train_modal_v5.py --stage 2    # item MLP + model head
modal run train/train_modal_v5.py --stage tune # ensemble weight tuning
```

### v11 — Approach 2: Lookup Table

A lightweight lookup table of empirical per-(subject, benchmark, condition) accuracy rates computed from the public training data. No GPU or model inference required.

**Building the lookup table:**

```bash
pip install datasets huggingface_hub
python train/compute_lookup.py
```

This writes `lookup.json` directly into `my_submission_v11_lookup/`.

## Validating Submissions

### Check a ZIP

```bash
python tools/check_submission_zip.py my_submission_v5_ensemble.zip
python tools/check_submission_zip.py my_submission_v11_lookup.zip
```

### Run a local smoke test against an unpacked directory

```bash
python tools/run_smoke_test.py my_submission_v5_ensemble/
python tools/run_smoke_test.py my_submission_v11_lookup/
```

The smoke test sets `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1`, which skips model loading in both submissions so you can verify the interface without downloading weights.

## Repository Layout

```
starting_kit/
├── my_submission_v5_ensemble/   # Approach 1 submission files
│   ├── model.py                 # predict() entry point
│   ├── labeling.py              # acquisition_function() entry point
│   ├── nn_weights.pt            # factor model + MLP weights
│   ├── subject_abilities.npy    # per-subject latent vectors
│   ├── subject_index.json       # subject name → index map
│   ├── ensemble_weights.json    # w_factor, w_llm, bias
│   └── models.txt               # LLM to pre-download (Qwen/Qwen3.5-9B)
├── my_submission_v11_lookup/    # Approach 2 submission files
│   ├── model.py                 # predict() entry point
│   ├── labeling.py              # acquisition_function() entry point
│   └── lookup.json              # empirical accuracy lookup table
├── train/
│   ├── train_modal_v5.py        # Modal orchestrator for v5 training
│   ├── train_factor_model_v5.py # Stage 1: factor model
│   ├── train_nn_v5.py           # Stage 2: item MLP + model head
│   ├── tune_ensemble_v5.py      # Stage tune: ensemble weights
│   └── compute_lookup.py        # Builds lookup.json for v11
└── tools/
    ├── check_submission_zip.py  # Validate a submission ZIP
    └── run_smoke_test.py        # Local smoke test against unpacked dir
```
