# Knowledge Editing Reversibility & Butterfly Effect Analysis

This repository contains the experimental framework developed for the Master's Thesis on **reversibility in Knowledge Editing (KE)** for Large Language Models (LLMs).  
The project investigates how factual edits applied to a model can be **applied, evaluated, and reverted**, while analyzing **side effects** on unrelated knowledge using Butterfly Effect–style metrics.

The framework is built on top of **EasyEdit** and supports editing methods such as **ROME** and **MEMIT**, with experiments conducted on autoregressive language models (e.g., GPT-2-XL, GPT-J).

---

## 1. Project Goals

The main objectives of this project are:

- Apply **localized factual edits** to pretrained language models
- Measure **edit effectiveness** on the target fact
- Analyze **collateral changes** on unrelated facts (Butterfly Effect)
- Study **edit reversibility**, i.e. whether a sequence of inverse edits can restore the original model behavior
- Provide a **clean, reproducible experimental pipeline** suitable for academic research

---

## 2. Background Concepts

### 2.1 Knowledge Editing

**Knowledge Editing** refers to techniques that modify a pretrained model’s internal representations in order to change specific factual knowledge **without retraining the model from scratch**.

Formally, given a model *M* and a factual statement *(s, r, o)*:
- Before editing:  
  `M(s, r) → o_old`
- After editing:  
  `M'(s, r) → o_new`

The goal is to enforce this change **while preserving all other knowledge**.

---

### 2.2 ROME (Rank-One Model Editing)

ROME edits a model by applying a **rank-one update** to a specific MLP layer.  
The update is computed so that the hidden representation corresponding to the edited fact is redirected toward the desired output.

Key properties:
- Single-layer intervention
- Fast and deterministic
- Highly localized, but still prone to side effects

---

### 2.3 MEMIT (Mass Editing Memory in Transformers)

MEMIT generalizes ROME to **multiple edits**, distributing updates across layers to store a set of new facts more robustly.

Compared to ROME:
- Supports batch edits
- Better retention across prompts
- Higher risk of global interference if not controlled

---

### 2.4 Butterfly Effect in Knowledge Editing

In Knowledge Editing, the **Butterfly Effect** refers to unintended changes in model behavior on **unrelated prompts** after a factual edit.

Even when an edit is successful locally, it may:
- Alter probabilities of unrelated tokens
- Change generations for semantically distant facts
- Affect linguistic fluency or coherence

This project explicitly measures these effects.

---

## 3. Repository Structure

```text
thesis_experiments/
│
├── configs/                # YAML experiment and hyperparameter configs
│   ├── suite_*_rome.yaml
│   ├── suite_*_memit.yaml
│   └── hparams_*.yaml
│
├── data/
│   └── me_ppl/
|       ├── ME-PPL_1k.json
|       └── ME-PPL_50.json
│
├── scripts/
│   ├── run_single_edit.py  # Single edit experiment
│   ├── run_edit_suite.py   # Batch experiment runner
│   ├── measure_butterfly.py
│   └── dataset_*.py
│
├── results/
│   ├── logs/
│   ├── metrics/
│   └── generations/
│
└── README.md
