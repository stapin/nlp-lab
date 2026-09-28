"""Shared paths, label definitions, data loading and metrics."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FEATS = ROOT / "features"
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"

HF_REPO = "iis-research-team/AINL-Eval-2025"
SEED = 42

KNOWN = ["human", "gpt-4-turbo", "gemma-2-27b", "llama-3.3-70b"]
UNKNOWN = "unknown"
LABELS5 = KNOWN + [UNKNOWN]

# The official prompt from the AINL-Eval 2025 paper (Sec. 3.1).
OFFICIAL_PROMPT = (
    "Сгенерируй краткое содержание научной статьи по заголовку и ключевым словам. "
    "Напиши только текст аннотации. Не начинай текст аннотации с фразы \"В данной статье\".\n"
    "Заголовок: {title}.\n"
    "Ключевые слова: {keywords}"
)

# Generators we add ourselves. "aux" texts are used as pseudo-unknown training data,
# "heldout" texts are only used for evaluation (never seen in training).
AUX_GENERATORS = {
    "qwen2.5-3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen3-4b-2507": "Qwen/Qwen3-4B-Instruct-2507",
}
HELDOUT_GENERATORS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "yandexgpt-5-lite-8b": "yandex/YandexGPT-5-Lite-8B-instruct",
}


def load_split(name: str) -> pd.DataFrame:
    """name in {train, val, dev, aux, heldout}; see prepare_data.py / generate.py."""
    return pd.read_parquet(DATA / f"{name}.parquet")


def truncate_words(texts, k):
    if not k:
        return list(texts)
    return [" ".join(t.split()[:k]) for t in texts]


def multiclass_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true))),
    }


def binary_metrics(is_ai_true, ai_score, threshold=0.5):
    is_ai_true = np.asarray(is_ai_true).astype(int)
    ai_score = np.asarray(ai_score, dtype=float)
    pred = (ai_score >= threshold).astype(int)
    out = {
        "acc": accuracy_score(is_ai_true, pred),
        "f1": f1_score(is_ai_true, pred),
    }
    if len(set(is_ai_true)) > 1:
        out["auroc"] = roc_auc_score(is_ai_true, ai_score)
    return out


def tpr_at_fpr(is_ai_true, ai_score, fpr=0.05):
    """Recall on AI texts when the threshold lets through `fpr` of human texts."""
    is_ai_true = np.asarray(is_ai_true).astype(bool)
    ai_score = np.asarray(ai_score, dtype=float)
    thr = np.quantile(ai_score[~is_ai_true], 1 - fpr)
    return float((ai_score[is_ai_true] > thr).mean())


def eval_splits(tag):
    """Splits a trained model is applied to. heldout exists only after generate.py --assemble."""
    splits = ["val", "dev"]
    if all((FEATS / tag / f"heldout_{g}.npy").exists() for g in ["style", "tfidf", "lm", "emb"]):
        splits.append("heldout")
    return splits


def save_preds(method, tag, split_probs, classes):
    """split_probs: {split: prob matrix [n, len(classes)]}. Writes results/preds/<method>__<tag>.parquet."""
    rows = []
    for split, probs in split_probs.items():
        df = load_split(split)[["label"]].copy()
        if "generator" in load_split(split):
            df["generator"] = load_split(split)["generator"].values
        df["split"] = split
        for j, c in enumerate(classes):
            df[f"p_{c}"] = probs[:, j]
        rows.append(df)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_parquet(RESULTS / "preds" / f"{method}__{tag}.parquet")


def train_frame(use_aux, trunc=0):
    """Training texts/labels: train split, optionally plus our pseudo-unknown generations."""
    df = load_split("train")
    if use_aux:
        df = pd.concat([df, load_split("aux")], ignore_index=True)
    return truncate_words(df.text, trunc), df.label.values
