# Knowledge Editing Reversibility & Butterfly Effect Analysis

This repository contains the experimental framework developed for the Master's Thesis on **reversibility in Knowledge Editing (KE)** for Large Language Models (LLMs).
The project investigates how factual edits applied to a model can be **applied, evaluated, and reverted**, while analyzing **side effects** on unrelated knowledge using Butterfly Effect–style metrics.

The framework is built on top of **EasyEdit** and supports editing methods such as **ROME** and **MEMIT**, with experiments conducted on autoregressive language models (e.g., GPT-2-XL, GPT-J).

---

## 1. Project Goals

The main objectives of this project are:

* Apply **localized factual edits** to pretrained language models
* Measure **edit effectiveness** on the target fact
* Analyze **collateral changes** on unrelated facts (Butterfly Effect)
* Study **edit reversibility**, i.e. whether a sequence of inverse edits can restore the original model behavior
* Provide a **clean, reproducible experimental pipeline** suitable for academic research

---

## 2. Background Concepts

### 2.1 Knowledge Editing

**Knowledge Editing** refers to techniques that modify a pretrained model’s internal representations in order to change specific factual knowledge **without retraining the model from scratch**.

Formally, given a model *M* and a factual statement *(s, r, o)*:

* Before editing:
  `M(s, r) → o_old`
* After editing:
  `M'(s, r) → o_new`

The goal is to enforce this change **while preserving all other knowledge**.

---

### 2.2 ROME (Rank-One Model Editing)

ROME edits a model by applying a **rank-one update** to a specific MLP layer.
The update is computed so that the hidden representation corresponding to the edited fact is redirected toward the desired output.

Key properties:

* Single-layer intervention
* Fast and deterministic
* Highly localized, but still prone to side effects

---

### 2.3 MEMIT (Mass Editing Memory in Transformers)

MEMIT generalizes ROME to **multiple edits**, distributing updates across layers to store a set of new facts more robustly.

Compared to ROME:

* Supports batch edits
* Better retention across prompts
* Higher risk of global interference if not controlled

---

### 2.4 Butterfly Effect in Knowledge Editing

In Knowledge Editing, the **Butterfly Effect** refers to unintended changes in model behavior on **unrelated prompts** after a factual edit.

Even when an edit is successful locally, it may:

* Alter probabilities of unrelated tokens
* Change generations for semantically distant facts
* Affect linguistic fluency or coherence

This project explicitly measures these effects.

---

## 3. Repository Structure (to be updated)

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
└── scripts/
    ├── run_single_edit.py  # Single edit experiment
    ├── butterfly_effect.py
    └── dataset_*.py
      └── old/
          └── run*.py
```

---

## 4. Getting Started (Linux Only)

> **Important note**
> This setup guide currently targets **Linux systems only**.
> A dedicated guide for **Windows 11** will be provided in a future update.

### Index

* [4.1 Install Anaconda / Miniconda](#41-install-anaconda--miniconda)
* [4.2 Obtain the Repository](#42-obtain-the-repository)
* [4.3 Create the Python Environment](#43-create-the-python-environment)
* [4.4 Configure the Experiment (YAML)](#44-configure-the-experiment-yaml)
* [4.5 Run the Reference Script](#45-run-the-reference-script)
* [4.6 Logs and Results](#46-logs-and-results)

---

### 4.1 Install Anaconda / Miniconda

To run the framework, you need a Conda-based Python distribution.

* **Recommended**: Miniconda (lighter and easier to manage)
* Alternative: Full Anaconda distribution

Follow the official instructions for Linux:

* [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

After installation, ensure that `conda` is available in your shell:

```bash
conda --version
```

---

### 4.2 Obtain the Repository

You have two supported options.

#### Option A — Clone from GitHub (recommended)

```bash
git clone https://github.com/emanuele-caddeo/knowledge-editing-reversibility-with-EasyEdit.git
cd knowledge-editing-reversibility-with-EasyEdit
```

#### Option B — Use a ZIP archive

If you already have the repository as a ZIP file:

```bash
unzip knowledge-editing-reversibility-with-EasyEdit.zip
cd knowledge-editing-reversibility-with-EasyEdit
```

---

### 4.3 Create the Python Environment

Create a dedicated Conda environment (Python **3.10 is strongly recommended**):

```bash
conda create -n easyedit python=3.10
conda activate easyedit
```

Install all required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

> **Note**: Depending on your hardware and CUDA setup, PyTorch may install either CPU or GPU builds.

---

### 4.4 Configure the Experiment (YAML)

The framework is fully configured through **YAML files**.

Experiment-level configuration files are located in:

```text
thesis_experiments/configs/
```

As a reference example, inspect and adapt:

```text
thesis_experiments/configs/exp_counterfact_hf_llama32_3b_rome_single.yaml
```

This file controls:

* Dataset source (local vs HuggingFace)
* Model selection
* Editing method (ROME / MEMIT)
* Butterfly Effect (PPL) evaluation

Make sure paths inside the YAML follow **Linux-style conventions**, e.g.:

```yaml
exp_local_dataset: data/counterfact/test.json
```

---

### 4.5 Run the Reference Script

From the project root, launch the main experiment script:

```bash
python -m thesis_experiments.scripts.run_single_edit_with_BE \
  --config thesis_experiments/configs/exp_counterfact_hf_llama32_3b_rome_single.yaml \
  --mode both
```

#### First run note (ROME statistics)

On the **first execution**, if the statistics file required by ROME is **not already present** (e.g. for a given `mom2_n_samples` value specified in the YAML), the framework will automatically compute it.

This step:

* Can take **a long time**
* Becomes especially expensive for large values (e.g. `mom2_n_samples = 10000`)

This is expected behavior and only happens once per configuration.

---

### 4.6 Logs and Results

During and after execution, you will see detailed logs printed to stdout.

All experiment artifacts are saved under:

```text
logs/
```

Each run creates a timestamped subdirectory containing:

* Execution logs
* Metrics (`results.json`)
* Butterfly Effect reports

These outputs are the primary data used for analysis and for the thesis evaluation.

---
