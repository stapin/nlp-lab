"""Download AINL-Eval 2025 and build train / val / dev splits.

train, val: stratified 90/10 split of the official train set (4 classes).
dev: the official development set with gold labels (5 classes, `unknown` = GigaChat-Lite).
The official private test set has no public labels, so dev is our main benchmark.
"""
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split

from common import DATA, HF_REPO, SEED


def read(fname):
    df = pd.read_csv(hf_hub_download(HF_REPO, fname, repo_type="dataset"))
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    df = df[df["text"].str.len() > 0].copy()
    df["label"] = df["label"].replace({"abstract": "human"})
    return df.reset_index(drop=True)


def main():
    DATA.mkdir(exist_ok=True)
    train = read("train.csv")
    dev = read("dev_full.csv")
    train, val = train_test_split(train, test_size=0.1, stratify=train["label"], random_state=SEED)
    for name, df in [("train", train), ("val", val), ("dev", dev)]:
        df = df.reset_index(drop=True)
        df["split"] = name
        df.to_parquet(DATA / f"{name}.parquet")
        print(name, len(df), df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
