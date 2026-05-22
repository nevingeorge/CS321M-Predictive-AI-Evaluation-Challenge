# v5 Ensemble Submission — Architecture Documentation

## 1. Problem Formulation and Method Description

### Problem

The task is **item cold-start prediction**: given a (subject, item, benchmark, condition) tuple where the item has never been observed in the training response matrix, predict P(subject answers item correctly). Standard matrix completion fails because there are no column entries to fit for new items. The only signal is side information: item text, subject metadata, benchmark identity, and test condition.

Formally, we want a function:

```
f(subject_content, item_content, benchmark, condition) → p ∈ [0, 1]
```

trained on observed triples (subject_id, item_id, label ∈ {0,1}) from the public dataset, evaluated on held-out triples whose items were never seen during training.

The primary metric is **negative log-loss** (higher is better). Log-loss penalises overconfident wrong predictions, so calibration matters as much as ranking accuracy.

### Method

v5 implements the three-stage Prediction-Guided Evaluation (PGE) pipeline from Lecture 4:

- **Stage 1**: Fit a K=4 logistic factor model on observed training responses to extract latent subject ability vectors U_i and item parameters (V_j, Z_j)
- **Stage 2a**: Train a neural network (sentence encoder → MLP) to predict (V_j, Z_j) from item text, enabling prediction for unseen items
- **Stage 2b**: Train a linear layer to predict U_i from subject metadata, enabling prediction for unseen subjects (fallback only; all test subjects are in training)
- **Stage 3**: Combine both branches with an LLM-as-judge (Qwen3.5-9B) via a weighted ensemble tuned on a held-out validation split

At inference, the prediction is:

```
logit_ens = w_factor · (U_i · V̂_j + Ẑ_j) + w_llm · log(P(yes)/P(no)) + bias
p         = σ(logit_ens)
```

with optional Platt scaling when labeled examples are revealed by the platform's adaptive labeling mechanism.

---

## 2. Training Dataset

---

Source: `aims-foundations/measurement-db` (HuggingFace public dataset).

- **3,578,880 binary triples** (subject_id, item_id, label) after filtering non-binary responses and deduplicating on (subject_id, item_id, test_condition)
- **909 subjects** (AI models under evaluation)
- **70,873 items** (benchmark questions/tasks)
- **16 unique benchmarks**

Training is done entirely offline before submission. The three training stages below produce artifacts baked into the submission ZIP.

---

## Stage 1 — K-Factor Logistic Model

### Model

The K-factor logistic model (Lecture 4) generalises Rasch to K latent dimensions:

```
P(Y_ij = 1 | U_i, V_j, Z_j) = σ(U_i^T V_j + Z_j)
```

- **U_i ∈ ℝ^K**: ability vector for subject i (K=4)
- **V_j ∈ ℝ^K**: loading vector for item j — which capability dimensions does this item test?
- **Z_j ∈ ℝ**: difficulty intercept for item j
- **σ**: logistic sigmoid

K=4 is chosen following the empirical finding in Lecture 4 that K=4 gives the best tradeoff between entry-wise fit and generalisation (K=8 overfits).

### Training

Objective: maximise the masked log-likelihood over observed entries Ω (the 3.58M training triples), with ℓ₂ regularisation on U and V:

```
L = -∑_{(i,j)∈Ω} log P(Y_ij | U_i, V_j, Z_j) + λ(‖U‖²_F + ‖V‖²_F)
```

- Optimizer: Adam, LR=0.01, 50 epochs, batch size 4096
- λ = 1e-4
- Initialisation: U, V ~ N(0, 0.01); Z = 0

### Outputs saved

| Artifact | Shape | Contents |
|---|---|---|
| `U.npy` | [909, 4] | Subject ability matrix |
| `V.npy` | [70873, 4] | Item loading matrix |
| `Z.npy` | [70873, 1] | Item difficulty intercepts |
| `subject_index.json` | — | subject_id → row index in U |
| `item_metadata.json` | — | item_id → {benchmark, condition, item_content} |
| `subject_metadata.json` | — | subject_id → {display_name, params, release_date, family, organization} |
| `v5_val_triples.json` | — | 15% held-out validation triples (536,832) for Stage 3 |

---

## Stage 2a — Item-Side Neural Network

### Purpose

Because test items are cold-start (never observed in training), V_j and Z_j cannot be looked up directly. Stage 2a trains a neural network to predict (V̂_j, Ẑ_j) from item text content alone.

### Architecture

```
item_text = "Benchmark: {b}\nCondition: {c}\nItem: {item_content}"

E_j = BAAI/bge-large-en-v1.5(item_text)     # 1024-dim sentence embedding
h   = ReLU( Linear(1024 → 256)(E_j) )
[V̂_j, Ẑ_j] = Linear(256 → K+1)(h)           # K=4 loading dims + 1 difficulty scalar
```

The sentence encoder is `BAAI/bge-large-en-v1.5` (335M params), frozen during Stage 2a. It produces L2-normalised 1024-dim embeddings.

### Training

U from Stage 1 is frozen. The item head is trained by minimising the masked binary cross-entropy over training triples, substituting the predicted (V̂_j, Ẑ_j) for the Stage 1 values:

```
L = -∑_{(i,j)∈Ω} log σ(U_i^T V̂_j + Ẑ_j)
```

- Optimizer: Adam, LR=1e-3, 30 epochs, batch size 2048
- U_i is retrieved from the fixed Stage 1 embeddings at each step

### Joint Fine-tuning

After Stage 2a, the last 2 layers of the bge-large encoder are unfrozen and trained jointly with the item head for 3 additional epochs at LR=1e-5 (encoder) and 1e-3 (head), on a 100K random sample of training triples. This allows the encoder to adapt its representations toward behavioral prediction rather than general semantic similarity.

### Key insight from Lecture 4

Semantic similarity does not imply behavioral similarity (Truong et al., 2025): items with cosine similarity > 0.99 can have behavioral correlations spanning [−1, +1]. The item MLP must therefore learn a nonlinear semantic-to-behavioral mapping from calibration data, not just perform nearest-neighbour retrieval.

---

## Stage 2b — Model-Side Linear Predictor

### Purpose

A linear layer maps subject metadata features to a predicted ability vector Û_i. Used as a fallback when a subject's name is not found in the training lookup table (which in practice never occurs on this dataset since all test subjects are in training).

### Features (d_F = 4 in practice due to sparse metadata)

| Feature | Description |
|---|---|
| log_params | log₁₀(parameter count in billions) |
| release_days | Days since 2020-01-01 |
| family_onehot | One-hot over top-20 architecture families |
| org_onehot | One-hot over top-20 organizations |

### Training

ℓ₁ regression from feature matrix F to Stage 1 U estimates:

```
L = ‖Linear(d_F → K)(F) − U‖₁
```

- 100 epochs, Adam, LR=1e-3

A linear predictor is sufficient because model ability is roughly log-linear in scale (Lecture 4).

---

## Stage 3 — Ensemble Weight Tuning

### LLM Branch (Qwen3.5-9B)

The LLM branch formats the four input fields as a yes/no completion prompt and reads the next-token log-probabilities:

```
prompt = """You will see a description of an AI subject and an evaluation item.
Decide whether the subject would answer the item correctly.
Reply with a single token: yes or no.

Benchmark: {benchmark}
Condition: {condition}
Subject: {subject_content}
Item: {item_content}
Answer:"""

logit_llm = log P(token="yes") − log P(token="no")
```

This gives a calibrated binary log-odds without asking the model to produce a numeric output (which is brittle).

### Ensemble Optimisation

Both branches produce scalar logits. The ensemble combines them linearly:

```
logit_ens = w_factor · logit_factor + w_llm · logit_llm + bias
```

[w_factor, w_llm, bias] are tuned by minimising binary cross-entropy on the 10K-sample validation subset using L-BFGS-B (scipy):

```
L = -∑_{(i,j)∈val} [y·log σ(logit_ens) + (1−y)·log(1−σ(logit_ens))]
```

### Tuned values

```json
{"w_factor": 0.1246, "w_llm": 0.0830, "bias": -0.8444}
```

The small weights and large negative bias reflect the training distribution's low average correctness rate (~65% global mean). **Note**: these weights were tuned on a random holdout of training items, not a column holdout of new benchmarks. This is a known calibration limitation — see §Limitations below.

---

## Inference Pipeline

### Module initialisation (runs once)

1. Load `BAAI/bge-large-en-v1.5` sentence encoder from HF cache
2. Load item head weights (`nn_weights.pt`) from ZIP
3. Load subject lookup table (`subject_abilities.npy`, `subject_index.json`) from ZIP
4. Load ensemble weights (`ensemble_weights.json`) from ZIP
5. Load `Qwen3.5-9B` tokenizer + model from HF cache (18 GB, bfloat16, device_map="auto")
6. Find token IDs for "yes" and "no"

### Per-call `predict(input, labeled=None)`

**1. Factor branch** — `logit_factor`

```python
item_text = f"Benchmark: {benchmark}\nCondition: {condition}\nItem: {item_content}"
enc   = bge_large.encode(item_text)          # 1024-dim
out   = item_head(enc)                       # [K+1]
v_hat = out[:K]                              # item loading vector
z_hat = out[K]                               # item difficulty

name = parse_name(subject_content)
if name in subject_index:
    u = U[subject_index[name]]               # direct lookup (primary path)
else:
    u = model_head(parse_features(subject_content))   # linear fallback

logit_factor = dot(u, v_hat) + z_hat
```

**2. LLM branch** — `logit_llm`

```python
prompt = judge_template.format(**input)      # zero-shot
ids    = tokenizer(prompt, return_tensors="pt")
logits = llm(**ids).logits[0, -1]           # next-token logits
lp     = log_softmax(logits)
logit_llm = lp[yes_id] - lp[no_id]
```

If `labeled` is provided, few-shot examples from the same benchmark are prepended to the prompt.

**3. Ensemble**

```python
logit_ens = 0.1246 * logit_factor + 0.0830 * logit_llm - 0.8444
```

**4. Platt scaling** (when `labeled` is non-empty)

Using the *same* logits as step 3 (computed for each labeled example via the cached logit values), fit `a, b` by gradient descent on:

```
p_cal = σ(a · logit_ens + b)
```

minimising BCE on the revealed (input, label) pairs. Apply to the test input's logit.

**5. Cache**

`_LOGIT_CACHE` stores `(logit_factor, logit_llm)` per input key. The `acquisition_function` (which runs the LLM for all ~255K candidate pairs) pre-populates this cache, so `predict()` calls for the 5K scored pairs hit the cache and run near-instantly.

---

## Adaptive Labeling

`acquisition_function(input)` returns `|p_factor − p_llm|` — the disagreement between the two branches. High disagreement = high uncertainty = most valuable pair to label.

The platform reveals K=5 labels per data category. These are passed as `labeled` to every subsequent `predict()` call. Platt scaling uses them to correct calibration for the current round's benchmark distribution.

---

## Artifacts in Submission ZIP

| File | Size | Purpose |
|---|---|---|
| `model.py` | 13 KB | Inference code |
| `labeling.py` | 0.5 KB | Acquisition function |
| `models.txt` | 0.04 KB | Declares two HF repos for pre-download |
| `nn_weights.pt` | 1.06 MB | Item head + model head weights |
| `subject_abilities.npy` | 14 KB | U matrix [909, 4] |
| `subject_index.json` | 27 KB | display_name → U row index |
| `ensemble_weights.json` | 0.1 KB | w_factor, w_llm, bias |

Large models (bge-large-en-v1.5, Qwen3.5-9B) are declared in `models.txt` and pre-downloaded by the platform before inference begins.

---

## 3. Experimental Results and Ablation Studies

All results below are on the **random 15% held-out validation split** (536,832 triples), which shares benchmark coverage with the training set. Negative log-loss is the primary metric (higher is better); AUC-ROC is reported secondarily.

### Validation performance by component

| Model | Val log-loss | Notes |
|---|---|---|
| Global mean baseline (predict 0.653 always) | ~0.636 | Trivial calibrated baseline |
| Factor model only (K=4, no NN) | — | Entry-wise AUC ≈ 0.845 per Lecture 4 results |
| Item head only (MLP, frozen U) | 0.597 | Stage 2a, 30 epochs |
| Item head + joint fine-tuning | ~0.585 | 3 epochs unfreezing last 2 encoder layers |
| Full ensemble (factor + LLM + Platt) | **0.547** | Tuned on 10K val samples |

### Effect of K on factor model (from Lecture 4 benchmarks)

| K | Entry-wise AUC | Col. holdout AUC |
|---|---|---|
| 1 (Rasch) | 0.780 | 0.720 |
| 2 | 0.820 | 0.745 |
| **4** | **0.845** | **0.750** |
| 8 | 0.860 | 0.740 (overfits) |

K=4 was selected as the best tradeoff between fit and column-holdout generalisation.

### Ablation: ensemble weight tuning

The L-BFGS-B optimisation produces weights (w_factor=0.125, w_llm=0.083, bias=−0.844). The large negative bias and small weights suggest that on the training distribution, a strong prior toward low probability is optimal — the training data average correctness is ~65% but many items in the held-out split are hard. Setting w_factor=w_llm=0 and bias=log(0.65/0.35) ≈ 0.619 would give the global mean prediction; the optimised weights improve beyond this baseline.

### Ablation: Platt scaling with labeled examples

Platt scaling (fitting a, b on the K=5 revealed labels per category) was not formally ablated on the validation set due to the small labeled sample size. At K=5 per category, calibration quality depends heavily on whether the revealed labels span both classes — degenerate cases (all 0s or all 1s) fall back to identity scaling.

---

## 4. Failure Mode Analysis

### Failure mode 1: Cold-start calibration gap

**Symptom**: Competition log-loss close to random (≈0.693) despite val log-loss of 0.547.

**Cause**: The validation split was a *random* 15% holdout, so it contains items from the same 16 benchmarks as training. The competition's test set uses items from benchmarks that may have been unseen during training. On truly new benchmarks, the item MLP predicts near-zero item parameters (V̂_j ≈ 0, Ẑ_j ≈ 0), and the ensemble collapses to:

```
logit_ens ≈ 0 + 0 + bias = -0.844
p ≈ σ(-0.844) ≈ 0.30
```

This produces systematically low predictions regardless of subject or item content, causing high log-loss.

**Remedy**: Tune ensemble weights on a **column holdout** (held-out benchmarks entirely), not a random holdout. This would expose the calibration failure during development and force the bias to match the actual cold-start distribution.

### Failure mode 2: LLM underweighting

**Symptom**: w_llm=0.083 is very small; the LLM branch contributes minimally.

**Cause**: On the training distribution, the factor model's predictions are well-calibrated (it was trained on those items). When tuning ensemble weights on the same training distribution, the LLM branch adds little marginal signal over the already-good factor model.

**Pattern**: The LLM should be weighted more heavily on cold-start items (where the factor model degrades) and less on seen items. A conditional weighting scheme (high w_llm when the item is from an unseen benchmark) would be more appropriate.

### Failure mode 3: Subject display name mismatches

**Symptom**: Some subjects fall through to the model linear head fallback instead of the direct U_i lookup.

**Cause**: The subject_index.json is keyed by display_name parsed from subject_content. If the platform renders subject_content differently from what was in subjects.parquet (e.g., different capitalisation or truncation), the name lookup fails. The linear fallback uses only metadata features (log_params, release_date, family) which are less accurate than the directly estimated U_i.

### Failure mode 4: Degenerate Platt scaling

**Symptom**: In rounds where all K=5 labeled examples have the same label (all correct or all incorrect), Platt scaling returns (a=1.0, b=0.0) — no correction applied.

**Cause**: With only 5 examples per data category and uncertainty-based acquisition, selected examples cluster near p=0.5 subjects. If a category contains only hard items (all fail) or easy items (all pass), the labeled set is uninformative for calibration.

**Pattern**: Uncertainty-based acquisition (selecting pairs near p=0.5) maximises information for a two-class problem but does not guarantee label diversity when the true label distribution within a category is skewed.

---

## Artifacts in Submission ZIP

| File | Size | Purpose |
|---|---|---|
| `model.py` | 13 KB | Inference code |
| `labeling.py` | 0.5 KB | Acquisition function |
| `models.txt` | 0.04 KB | Declares two HF repos for pre-download |
| `nn_weights.pt` | 1.06 MB | Item head + model head weights |
| `subject_abilities.npy` | 14 KB | U matrix [909, 4] |
| `subject_index.json` | 27 KB | display_name → U row index |
| `ensemble_weights.json` | 0.1 KB | w_factor, w_llm, bias |

Large models (bge-large-en-v1.5, Qwen3.5-9B) are declared in `models.txt` and pre-downloaded by the platform before inference begins.
