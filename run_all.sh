#!/usr/bin/env bash
# Full pipeline. Runtime on 1x RTX 4090: ~15 min data collection + ~50 min features/models.
set -euo pipefail
cd "$(dirname "$0")/src"
PY=${PY:-python}

# 1. data
$PY prepare_data.py
$PY extract_meta.py                                        # title + keywords for 3.5k human abstracts
for g in qwen2.5-3b qwen3-4b-2507 qwen3-1.7b yandexgpt-5-lite-8b; do $PY generate.py --gen $g; done
$PY generate.py --assemble                                 # -> data/aux.parquet (pseudo-unknown), data/heldout.parquet

# 2. features: full texts and first-40-words (for the length-control experiment)
for T in 0 40; do
  for G in style tfidf lm emb; do
    $PY features.py --group $G --trunc $T --splits train val dev aux heldout
  done
done

# 3. models: TF-IDF baseline, zero-shot, mDeBERTa, hybrid — with/without pseudo-unknown, full/truncated
for T in 0 40; do
  TAG=$([ "$T" = 0 ] && echo full || echo trunc$T)
  $PY run_tfidf.py --trunc $T
  $PY run_tfidf.py --trunc $T --aux
  $PY run_zeroshot.py --tag $TAG
  $PY train_deberta.py --trunc $T
  $PY train_deberta.py --trunc $T --aux
  $PY run_hybrid.py --groups tfidf lm style --tag $TAG
  $PY run_hybrid.py --groups tfidf lm style --tag $TAG --aux
  $PY run_hybrid.py --groups tfidf emb lm style --tag $TAG
  $PY run_hybrid.py --groups emb lm style --tag $TAG
done
# feature-group ablation (full texts only)
for GR in "tfidf" "emb" "lm" "style" "lm style" "tfidf lm" "tfidf style"; do
  $PY run_hybrid.py --groups $GR
  $PY run_hybrid.py --groups $GR --aux
done
$PY run_hybrid.py --groups emb lm style --clf lgbm

# 4. evaluation, figures, LaTeX tables
$PY evaluate.py
$PY make_figures.py
$PY make_tables.py

# 5. compile the report (falls back to tectonic if pdflatex is not installed)
bash ../report/build_pdf.sh
