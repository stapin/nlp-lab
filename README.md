# Detecting AI-generated Russian scientific abstracts: length shortcuts and new generators

NLP course project built on the [AINL-Eval 2025](https://github.com/iis-research-team/AINL-Eval-2025) shared task
([paper](https://arxiv.org/abs/2508.09622)): given a Russian scientific abstract, decide whether it was written by a
human or by one of the LLMs `gpt-4-turbo`, `llama-3.3-70b`, `gemma-2-27b`, or by an **unknown** model.

What the project does:

1. **Reproduces the official baselines** and compares a fine-tuned mDeBERTa-v3, zero-shot detectors
   (log-perplexity, Binoculars) and a **hybrid feature-based detector** (TF-IDF stacking + Qwen3-0.6B token statistics +
   e5 embeddings + stylometry, linear classifier on top).
2. **Collects new data**: ~5.7k abstracts generated in 2025 by Qwen2.5-3B, Qwen3-4B-2507 and YandexGPT-5-Lite-8B with
   the official prompt (titles/keywords are reconstructed from human abstracts by an LLM).
   Two of the generators are used as *pseudo-unknown* training data, YandexGPT is kept as a held-out test generator.
3. **Length-control study**: human abstracts are ~2x longer than generated ones; we re-run every method on texts cut
   to the first 40 words to measure how much each detector relies on length.

The report is in [`report/main.tex`](report/main.tex) (PDF: `report/main.pdf`).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on a single RTX 4090 (24 GB). All models are downloaded from the Hugging Face Hub, no token required.

## Reproduce

```bash
bash run_all.sh            # everything: data -> generation -> features -> models -> metrics -> figures
```

or step by step (from `src/`):

| step | command | time (1x4090) |
|---|---|---|
| download + split data | `python prepare_data.py` | <1 min |
| reconstruct titles/keywords | `python extract_meta.py` | ~10 min |
| generate new abstracts | `python generate.py --gen <name>`, then `--assemble` | ~5-15 min per model |
| features | `python features.py --group {style,tfidf,lm,emb} [--trunc 40]` | lm ~9 min, emb ~2 min, others CPU |
| TF-IDF baseline | `python run_tfidf.py [--aux] [--trunc 40]` | ~1 min (CPU) |
| zero-shot | `python run_zeroshot.py [--tag trunc40]` | seconds |
| mDeBERTa-v3-base | `python train_deberta.py [--aux] [--trunc 40]` | ~4.5 min |
| hybrid | `python run_hybrid.py --groups tfidf lm style [--aux] [--tag trunc40]` | ~1 min (CPU) |
| metrics / figures | `python evaluate.py && python make_figures.py` | seconds |

Model training is only mDeBERTa (1 epoch) and linear classifiers; LLMs are used for inference only.

## Layout

```
src/
  common.py          paths, labels, metrics
  prepare_data.py    AINL-Eval -> data/{train,val,dev}.parquet
  extract_meta.py    title + keywords for human abstracts (Qwen2.5-3B-Instruct)
  generate.py        new generations with the official prompt -> data/{aux,heldout}.parquet
  features.py        style / tfidf / lm (Qwen3-0.6B, Binoculars) / emb (multilingual-e5-large)
  run_tfidf.py       official baseline
  run_zeroshot.py    log-perplexity and Binoculars thresholds
  train_deberta.py   fine-tuned mDeBERTa-v3-base
  run_hybrid.py      hybrid detector
  evaluate.py        all metrics -> results/metrics.csv
  make_figures.py    figures -> report/figures
data/                our generated data (aux, heldout, meta) is committed; AINL-Eval splits are re-downloaded
results/             metrics.csv, dataset_stats.csv
report/              LaTeX report
```

## Data and licences

AINL-Eval 2025 is distributed under Apache-2.0. Generated texts in `data/` were produced with openly licensed
models (Qwen: Apache-2.0; YandexGPT-5-Lite: YandexGPT-5-Lite-8B licence) and are released for research purposes.
