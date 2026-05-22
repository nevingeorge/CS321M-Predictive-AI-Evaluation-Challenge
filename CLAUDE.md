# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A starting kit for the **Predictive AI Evaluation Challenge** on Codabench. The task: given a benchmark, a test condition, a description of an AI model (`subject_content`), and an evaluation item (`item_content`), predict the probability (float in `[0, 1]`) that the AI model answers the item correctly. The primary metric is **negative log-loss** (higher is better); AUC-ROC is secondary.

**Submit at**: https://aimslab.stanford.edu/competition/submit  
**Final code record**: Gradescope (authoritative for grading if it differs from Codabench)

The hidden test set materializes 5000 items at runtime. Module-level code runs once at container startup; `predict()` is called once per `(model_id, item_id)` pair. **No internet access** at runtime except the organizer's internal data service. State does not persist across rounds (fresh container each time).

## The Core ML Problem

**Item cold-start prediction**: test items have *no* observed responses in the training matrix — they often come from benchmarks that didn't exist at training time. Test *subjects* are the same subjects seen in training, so the cold-start dimension is the item side alone. Standard matrix completion cannot help; the only signal is side information: item text, subject metadata, benchmark, and condition.

The recommended three-stage **Prediction-Guided Evaluation (PGE)** pipeline:
1. **Stage 1**: Extract latent ability estimates (Û_i) and item parameters from observed training responses (IRT, matrix factorization, etc.)
2. **Stage 2a** *(must generalize at test time)*: Learn a map from item text → item parameters
3. **Stage 2b** *(optional)*: Learn a map from subject metadata → ability; a simple lookup of Stage 1 ability estimates for known subjects is already a reasonable baseline
4. **Stage 3**: Combine at test time to produce a probability for any subject-item pair

## Grading

- **50% Technical report** (Gradescope): NeurIPS 2025 LaTeX template, max 4 pages of main content. Must cover: problem formulation, training data used, experimental results and ablation studies, and failure-mode analysis.
- **50% Leaderboard**: best submission score (negative log-loss, higher is better) across all rounds. Beating the organizer-provided baseline guarantees ≥80% on this component; ranking among teams that beat the baseline determines remaining points. Baseline score released 3 weeks after competition start.

Teams: 1–3 students. Teams lock at end of Week 2. Each team submits once per grading window; 50 scored submissions per team per UTC day, 1000 total.

## Local Development Commands

Create and validate a submission:
```bash
cp -R sample_code_submission my_submission
(cd my_submission && zip -r ../my_submission.zip .)

# Check ZIP layout (model.py must be at root, not nested in a folder)
python tools/check_submission_zip.py my_submission.zip

# Run smoke test against sample_data/ (catches interface and return-value errors)
python tools/run_smoke_test.py my_submission/
```

The smoke test sets `PREDICTIVE_EVAL_LOCAL_SMOKE_TEST=1` when importing your code. Use this env var in module-level code to skip expensive loading (e.g., HF model weights) during local testing.

## Required Interface

`model.py` must define:
```python
def predict(input: dict, labeled: list[dict] | None = None) -> float:
    ...
```

`input` keys: `benchmark`, `condition`, `subject_content`, `item_content` (all `str`).  
Return: a **native Python `float`** in `[0, 1]`. Wrap numpy/torch scalars: `return float(tensor.item())`. `NaN`, infinity, out-of-range, non-float types, or exceptions fail the submission.

The **same `labeled` list** is passed on every `predict()` call within a round. It will be empty or `None` in local smoke tests.

Optional `labeling.py` must define:
```python
def acquisition_function(input: dict) -> float:
    ...
```
Returns a native Python `float` (finite). Higher = higher labeling priority. Platform reveals top K=5 per data category and passes them to `predict()` as `labeled` (each dict has the same four keys plus `"label": int` in `{0, 1}`). Any bad return value (non-finite, exception, timeout) for *any* candidate discards all acquisition scores for the entire round and falls back to random selection.

`acquisition_function()` is called once per candidate with **no access to the full candidate list**. For cross-candidate strategies (diversity, k-center), accumulate state in module-level variables across calls within a round.

## Which Template To Use

| Situation | Use |
|---|---|
| No HuggingFace model needed | `sample_code_submission/` |
| Exactly one HF repo | `templates/hf_submission/` |
| Multiple HF repos (up to 5) | `templates/multi_hf_submission/` |
| Custom adaptive labeling | Add `templates/labeling_addon/labeling.py` to your submission |

## HuggingFace Models

Declare repos in `models.txt` (one per line, `#` comments ignored). Platform pre-downloads them before code runs and routes to GPU based on largest declared model. Use `local_files_only=True` when loading. Cache dir priority: `$HF_HOME` → `/app/hf_cache` → `.hf_cache/` next to `model.py`.

GPU tiers (B200-family, 8h timeout each): ≤70B params → B200, ≤140B → B200:2, ≤300B → B200:4. >300B for any single repo is rejected. Max 5 repos; combined download limit 1024 GB, 1000 GB per repo.

## Loading Training Data

The public dataset is at `aims-foundations/measurement-db` on HuggingFace. Do **not** use `load_dataset("aims-foundations/measurement-db")` directly — it may mix response and trace tables. Instead, load response tables explicitly, excluding `subjects.parquet`, `items.parquet`, `benchmarks.parquet`, and `*_traces.parquet`.

Response table schema: `subject_id`, `item_id`, `benchmark_id`, `trial`, `test_condition`, `response`, `correct_answer`, `trace`.

Key joins: response rows → items (by `item_id`) → subjects (by `subject_id`) → benchmarks (by `benchmark_id`). Use `benchmark["benchmark_id"]` (not `benchmark["name"]`) for the `benchmark` field passed to `predict()`. `subject_id` in training = `model_id` at runtime, but treat `subject_content` as display text only — do not parse it as a stable serialization.

Training data gotchas:
- Public responses may be binary, Likert, or fractional; hidden scoring uses binary labels. Filter/transform `response` values to match your training objective.
- Deduplicate on `(subject_id, item_id, test_condition)`, keeping the smallest `trial`.
- `test_condition` is part of the task context — preserve it in splits and aggregations.
- Participants are encouraged to curate additional training data beyond the provided dataset.

## Data Schema at Runtime

`subject_content` is a multiline string starting with `Name: <display_name>`, optionally followed by `Organization:`, `Parameters:`, `Released:`, `Family:` lines. Parse defensively; metadata lines may be absent.

## Modeling Approaches

**Content-based NCF baseline**: Embed `subject_content` and `item_content` with a sentence transformer (e.g., `all-mpnet-base-v2`, 768-dim), concatenate embeddings, pass through a small MLP head trained offline on the public dataset with BCE loss. Load encoder and MLP at module level. For adaptive use, fit one-parameter Platt scaling on the `labeled` dicts at the first call of each round and cache the calibrator as a module-level variable.

**LLM-as-judge baseline**: Format the four input fields into a prompt asking whether the subject would pass the item. Read the log-probabilities of `" yes"` / `" no"` next tokens and renormalize — do not ask the LLM for a numeric reply (models miscalibrate and emit non-numeric text). Declare the judge model in `models.txt`. For a 7B model against 5000 items, use batching, quantization (`bitsandbytes`, AWQ), or caching to fit within the round time budget. Use `labeled` dicts as few-shot examples bucketed by `"benchmark"`.

**k-means diversity acquisition**: Fit k-means on sentence embeddings of the public training set offline, save centroids. At test time assign each candidate to its nearest centroid, score by `-seen_count[cluster]` (first-in-cluster scores highest). Because `acquisition_function()` is stateless across calls, maintain a `Counter` at module level that is incremented each call. Alternative: online farthest-point sampling with a module-level list of already-encoded embeddings.

## Critical Implementation Notes

- **Load at module level, never inside `predict()`**: The container imports `model.py` once; `predict()` is called thousands of times. Loading a large checkpoint inside `predict()` would reload on every call and blow the time budget.
- **Wrap tensors**: `return float(tensor.item())` — numpy/torch scalars are not accepted and fail CSV serialization.
- **No state across rounds**: module-level variables are reset each round (fresh container). Do not attempt to accumulate labels or inputs across submissions.
- **Hosted APIs are fine offline**: during training data curation and model fitting, external APIs are allowed. Only `predict()` and `acquisition_function()` are network-isolated.

## Submission Constraints

- Training must be done offline. Bake small fitted state into the ZIP; large checkpoints go to HuggingFace declared in `models.txt`.
- `requirements.txt` support is organizer-controlled and defaults to disabled.
- `trust_remote_code` defaults to disabled.
- No nested ZIPs. `model.py` must be at the archive root.
- Each round scores a random subset of the 5000 hidden pairs (stratified by data category), preventing label recovery by repeated leaderboard probing.
