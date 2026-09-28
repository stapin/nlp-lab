"""Zero-shot binary detectors: log-perplexity and Binoculars (Hans et al., 2024) thresholds.

No classifier is trained; only the decision threshold is picked on the val split.
Scores are mapped to a pseudo-probability p_human so that the threshold corresponds to 0.5.
"""
import argparse
import json

import numpy as np

from common import FEATS, eval_splits, load_split, save_preds

NAMES = json.loads((FEATS / "lm_names.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()
    for feat in ["binoculars", "obs_logppl"]:
        col = NAMES.index(feat)
        score = {s: np.load(FEATS / args.tag / f"{s}_lm.npy")[:, col] for s in eval_splits(args.tag)}
        # low perplexity / low Binoculars score => machine-generated
        val_human = load_split("val").label.values == "human"
        cands = np.quantile(score["val"], np.linspace(0.01, 0.99, 197))
        accs = [((score["val"] > t) == val_human).mean() for t in cands]
        thr, scale = cands[int(np.argmax(accs))], score["val"].std()
        out = {}
        for s, v in score.items():
            p_h = 1 / (1 + np.exp(-(v - thr) / (0.1 * scale)))
            out[s] = np.stack([p_h, 1 - p_h], 1)
        save_preds(f"zeroshot-{feat}", args.tag, out, ["human", "ai"])


if __name__ == "__main__":
    main()
