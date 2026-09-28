"""Compute all metrics from results/preds/*.parquet -> results/metrics.csv.

Decision rules
  5-class (official AINL-Eval): models trained with an explicit `unknown` class use argmax;
      4-class models predict `unknown` when max prob < 0.6 (rule of the official baseline).
  binary: ai_score = 1 - p_human, threshold 0.5.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import KNOWN, RESULTS, UNKNOWN, binary_metrics, multiclass_metrics, tpr_at_fpr

OPEN_SET_THRESHOLD = 0.6


def decide(df):
    pcols = [c for c in df.columns if c.startswith("p_")]
    classes = [c[2:] for c in pcols]
    p = df[pcols].values
    pred = np.array(classes)[p.argmax(1)]
    if UNKNOWN not in classes:
        pred = np.where(p.max(1) < OPEN_SET_THRESHOLD, UNKNOWN, pred)
    return pred


def evaluate(df):
    res = {}
    multiclass = "p_ai" not in df.columns
    for split, d in df.groupby("split"):
        ai_true = (d.label != "human").values
        ai_score = 1 - d["p_human"].values
        b = binary_metrics(ai_true, ai_score)
        res.update({f"{split}_bin_{k}": v for k, v in b.items()})
        if split == "heldout":
            res["heldout_tpr@5fpr"] = tpr_at_fpr(ai_true, ai_score, 0.05)
            pred = decide(d) if multiclass else None
            if multiclass:
                res["heldout_unknown_recall"] = (pred[ai_true] == UNKNOWN).mean()
            for g in d.generator.unique():
                if g == "human":
                    continue
                keep = (d.generator.isin([g, "human"])).values
                res[f"heldout[{g}]_auroc"] = roc_auc_score(ai_true[keep], ai_score[keep])
                res[f"heldout[{g}]_tpr@5fpr"] = tpr_at_fpr(ai_true[keep], ai_score[keep], 0.05)
                if multiclass:
                    res[f"heldout[{g}]_unknown_recall"] = (pred[(d.generator == g).values] == UNKNOWN).mean()
            continue
        if multiclass:
            pred = decide(d)
            m = multiclass_metrics(d.label.values, pred)
            res.update({f"{split}_{k}": v for k, v in m.items()})
            if split == "dev":
                known = d.label.isin(KNOWN).values
                closed = np.array([c[2:] for c in d.columns if c.startswith("p_") and c[2:] in KNOWN])
                pk = d[[f"p_{c}" for c in closed]].values[known]
                res["dev_closed_acc"] = (closed[pk.argmax(1)] == d.label.values[known]).mean()
                res["dev_unknown_recall"] = (pred[~known] == UNKNOWN).mean()
    return res


def main():
    rows = []
    for f in sorted((RESULTS / "preds").glob("*.parquet")):
        method, tag = f.stem.split("__")
        rows.append({"method": method, "tag": tag, **evaluate(pd.read_parquet(f))})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "metrics.csv", index=False)
    cols = ["method", "tag", "dev_acc", "dev_macro_f1", "dev_closed_acc", "dev_unknown_recall",
            "dev_bin_acc", "dev_bin_auroc", "heldout_bin_auroc", "heldout_tpr@5fpr", "heldout_unknown_recall"]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
