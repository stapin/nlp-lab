"""Feature extraction for the hybrid detector.

Three feature groups, each saved as features/<tag>/<split>_<group>.npy:
  style -- hand-crafted stylometric statistics (CPU)
  lm    -- token-level statistics from two small LMs (Qwen3-0.6B-Base = observer,
           Qwen3-0.6B = performer), incl. the Binoculars score
  emb   -- mean-pooled multilingual-e5-large sentence embeddings
  tfidf -- class probabilities of the TF-IDF + LR baseline (5-fold out-of-fold on train), i.e. stacking

  python features.py --group lm --splits train val dev [--trunc 40]
"""
import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import functools
import json
import re
import string

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from common import FEATS, load_split, truncate_words

OBSERVER = "Qwen/Qwen3-0.6B-Base"
PERFORMER = "Qwen/Qwen3-0.6B"
EMBEDDER = "intfloat/multilingual-e5-large"

SENT_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"\w+", re.U)
PUNCT = set(string.punctuation + "«»—–…")

STYLE_NAMES = ["n_words", "n_chars", "n_sents", "mean_word_len", "mean_sent_len", "std_sent_len",
               "digit_frac", "latin_frac", "upper_frac", "punct_frac", "comma_per_word",
               "paren_per_word", "dash_per_word", "ttr"]
LM_NAMES = ["obs_logppl", "obs_logppl_std", "obs_entropy", "obs_rank1", "obs_rank10", "obs_logrank",
            "perf_logppl", "perf_entropy", "perf_rank1", "xppl", "binoculars", "n_tokens"]


def style_features(text):
    words = WORD_RE.findall(text)
    sents = [s for s in SENT_RE.split(text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sents] or [0]
    nw, nc = max(len(words), 1), max(len(text), 1)
    letters = [c for c in text if c.isalpha()] or [" "]
    return [
        len(words), len(text), len(sents),
        np.mean([len(w) for w in words]) if words else 0.0,
        np.mean(sent_lens), np.std(sent_lens),
        sum(c.isdigit() for c in text) / nc,
        sum(c.isascii() for c in letters) / len(letters),
        sum(c.isupper() for c in letters) / len(letters),
        sum(c in PUNCT for c in text) / nc,
        text.count(",") / nw,
        text.count("(") / nw,
        (text.count("—") + text.count("–") + text.count(" - ")) / nw,
        len({w.lower() for w in words}) / nw,
    ]


@functools.cache
def load_lms():
    tok = AutoTokenizer.from_pretrained(OBSERVER)
    obs = AutoModelForCausalLM.from_pretrained(OBSERVER, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    perf = AutoModelForCausalLM.from_pretrained(PERFORMER, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    return tok, obs, perf


def token_batches(lengths, budget):
    """Length-sorted batches with at most `budget` padded tokens each (logits are vocab-sized)."""
    order = np.argsort(lengths)
    batch = []
    for j in order:
        if batch and (len(batch) + 1) * lengths[j] > budget:
            yield np.array(batch)
            batch = []
        batch.append(j)
    if batch:
        yield np.array(batch)


@torch.no_grad()
def lm_features(texts, budget=2048, max_len=512):
    tok, obs, perf = load_lms()
    ids = tok(list(texts), truncation=True, max_length=max_len)["input_ids"]
    out = np.zeros((len(texts), len(LM_NAMES)), dtype=np.float32)
    for idx in tqdm(list(token_batches([len(x) for x in ids], budget)), desc="lm"):
        enc = tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt").to("cuda")
        lo = obs(**enc).logits[:, :-1].float()
        lp = perf(**enc).logits[:, :-1].float()
        tgt = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].float()
        n = mask.sum(1).clamp(min=1)

        logp_o, logp_p = F.log_softmax(lo, -1), F.log_softmax(lp, -1)
        nll_o = -logp_o.gather(-1, tgt[..., None])[..., 0]
        nll_p = -logp_p.gather(-1, tgt[..., None])[..., 0]
        p_o = logp_o.exp()
        ent_o = -(p_o * logp_o).sum(-1)
        ent_p = -(logp_p.exp() * logp_p).sum(-1)
        xent = -(p_o * logp_p).sum(-1)  # cross-perplexity term of Binoculars
        tgt_logit_o = lo.gather(-1, tgt[..., None])
        rank_o = (lo > tgt_logit_o).sum(-1) + 1
        tgt_logit_p = lp.gather(-1, tgt[..., None])
        rank_p = (lp > tgt_logit_p).sum(-1) + 1

        def mean(x):
            return (x * mask).sum(1) / n

        obs_ppl, perf_ppl, xppl = mean(nll_o), mean(nll_p), mean(xent)
        std_o = torch.sqrt(mean((nll_o - obs_ppl[:, None]) ** 2))
        feats = torch.stack([
            obs_ppl, std_o, mean(ent_o), mean((rank_o == 1).float()), mean((rank_o <= 10).float()),
            mean(torch.log(rank_o.float())),
            perf_ppl, mean(ent_p), mean((rank_p == 1).float()), xppl, perf_ppl / xppl, n,
        ], 1)
        out[idx] = feats.cpu().numpy()
        del lo, lp, logp_o, logp_p, p_o
    return out


@functools.cache
def load_embedder():
    tok = AutoTokenizer.from_pretrained(EMBEDDER)
    return tok, AutoModel.from_pretrained(EMBEDDER, torch_dtype=torch.float16, device_map="cuda").eval()


@torch.no_grad()
def emb_features(texts, batch=128, max_len=512):
    tok, model = load_embedder()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    for i in tqdm(range(0, len(texts), batch), desc="emb"):
        idx = order[i:i + batch]
        enc = tok(["query: " + texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                  max_length=max_len).to("cuda")
        h = model(**enc).last_hidden_state.float()
        m = enc.attention_mask[..., None].float()
        e = F.normalize((h * m).sum(1) / m.sum(1), dim=-1)
        out[idx] = e.cpu().numpy()
    return out


def tfidf_features(split, trunc, folds=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    def fit(texts, labels):
        vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5)
        clf = LogisticRegression(C=5, max_iter=1000).fit(vec.fit_transform(texts), labels)
        return lambda t: clf.predict_proba(vec.transform(t))

    train = load_split("train")
    texts, labels = np.array(truncate_words(train.text, trunc), dtype=object), train.label.values
    if split == "train":
        out = np.zeros((len(texts), len(set(labels))), dtype=np.float32)
        for tr, te in StratifiedKFold(folds, shuffle=True, random_state=0).split(texts, labels):
            out[te] = fit(texts[tr], labels[tr])(texts[te])
        return out
    return fit(texts, labels)(truncate_words(load_split(split).text, trunc)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=["style", "lm", "emb", "tfidf"], required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "dev", "aux", "heldout"])
    ap.add_argument("--trunc", type=int, default=0, help="keep only the first K words (length control)")
    args = ap.parse_args()

    tag = f"trunc{args.trunc}" if args.trunc else "full"
    (FEATS / tag).mkdir(parents=True, exist_ok=True)
    names = {"style": STYLE_NAMES, "lm": LM_NAMES}.get(args.group)
    if names:
        (FEATS / f"{args.group}_names.json").write_text(json.dumps(names))
    for split in args.splits:
        texts = truncate_words(load_split(split).text, args.trunc)
        if args.group == "style":
            x = np.array([style_features(t) for t in tqdm(texts, desc=split)], dtype=np.float32)
        elif args.group == "tfidf":
            x = tfidf_features(split, args.trunc)
        elif args.group == "lm":
            x = lm_features(texts)
        else:
            x = emb_features(texts)
        np.save(FEATS / tag / f"{split}_{args.group}.npy", x)
        print(tag, split, args.group, x.shape)


if __name__ == "__main__":
    main()
