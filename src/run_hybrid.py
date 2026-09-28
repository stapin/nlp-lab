"""Hybrid detector: [e5 embeddings | LM statistics | stylometry] -> linear / GBDT classifier.

  python run_hybrid.py --groups emb lm style --clf lr [--aux] [--tag trunc40]
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import FEATS, SEED, eval_splits, load_split, save_preds


def load_x(split, groups, tag):
    parts = []
    for g in groups:
        x = np.load(FEATS / tag / f"{split}_{g}.npy")
        if g != "emb":  # heavy-tailed scalars (lengths, perplexities): signed log
            x = np.sign(x) * np.log1p(np.abs(x))
        parts.append(x)
    return np.nan_to_num(np.hstack(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["emb", "lm", "style"])
    ap.add_argument("--clf", choices=["lr", "lgbm"], default="lr")
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    x = load_x("train", args.groups, args.tag)
    y = load_split("train").label.values
    if args.aux:
        x = np.vstack([x, load_x("aux", args.groups, args.tag)])
        y = np.concatenate([y, load_split("aux").label.values])

    if args.clf == "lr":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    else:
        clf = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8,
                                 subsample_freq=1, colsample_bytree=0.5, random_state=SEED, verbose=-1)
    clf.fit(x, y)
    out = {s: clf.predict_proba(load_x(s, args.groups, args.tag)) for s in eval_splits(args.tag)}
    name = f"hybrid-{args.clf}[{'+'.join(args.groups)}]" + ("+aux" if args.aux else "")
    save_preds(name, args.tag, out, list(clf.classes_))


if __name__ == "__main__":
    main()
