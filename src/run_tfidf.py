"""TF-IDF + logistic regression, replicating the official AINL-Eval baseline."""
import argparse

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import eval_splits, load_split, save_preds, train_frame, truncate_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", action="store_true", help="add our generations as an explicit `unknown` class")
    ap.add_argument("--trunc", type=int, default=0)
    args = ap.parse_args()
    tag = f"trunc{args.trunc}" if args.trunc else "full"

    texts, labels = train_frame(args.aux, args.trunc)
    # same hyper-parameters as baseline/tfidf_baseline.py in the official repository
    vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
    clf = LogisticRegression(C=5, max_iter=1000)
    clf.fit(vec.fit_transform(texts), labels)
    out = {s: clf.predict_proba(vec.transform(truncate_words(load_split(s).text, args.trunc)))
           for s in eval_splits(tag)}
    save_preds("tfidf" + ("+aux" if args.aux else ""), tag, out, list(clf.classes_))


if __name__ == "__main__":
    main()
