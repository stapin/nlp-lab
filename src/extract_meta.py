"""Recover a title and keywords for human abstracts with an LLM.

AINL-Eval ships only abstract texts, but its generators were prompted with the paper
title and keywords. To reproduce the official generation protocol with new models we
first reconstruct (title, keywords) from a human abstract.

Sources:
  aux     -- human abstracts from our train split (their generations become pseudo-unknown training data)
  heldout -- human abstracts from dev (their generations form the held-out 2025-generators test set)
"""
import argparse
import json
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DATA, SEED, load_split

META_PROMPT = (
    "Ниже приведена аннотация научной статьи на русском языке. Придумай для неё правдоподобный "
    "заголовок статьи и 3-6 ключевых слов. Ответь строго в формате JSON без пояснений: "
    "{{\"title\": \"...\", \"keywords\": [\"...\", \"...\"]}}\n\nАннотация: {text}"
)


def parse(out):
    m = re.search(r"\{.*\}", out, flags=re.S)
    if not m:
        return None, None
    try:
        d = json.loads(m.group(0))
        kw = d.get("keywords", [])
        kw = ", ".join(kw) if isinstance(kw, list) else str(kw)
        title = str(d.get("title", "")).strip().rstrip(".")
        return (title or None), (kw or None)
    except json.JSONDecodeError:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_aux", type=int, default=2500)
    ap.add_argument("--n_heldout", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    train, dev = load_split("train"), load_split("dev")
    aux = train[train.label == "human"].sample(args.n_aux, random_state=SEED)
    held = dev[dev.label == "human"].sample(args.n_heldout, random_state=SEED)
    src = pd.concat([aux.assign(source="aux"), held.assign(source="heldout")], ignore_index=True)

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": META_PROMPT.format(text=t)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for t in src.text
    ]
    titles, kws = [], []
    for i in tqdm(range(0, len(prompts), args.batch)):
        enc = tok(prompts[i:i + args.batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=120, do_sample=False)
        for o in tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True):
            t, k = parse(o)
            titles.append(t)
            kws.append(k)
    src["title"], src["keywords"] = titles, kws
    ok = src.title.notna() & src.keywords.notna()
    print(f"parsed {ok.sum()}/{len(src)}")
    src[ok].reset_index(drop=True).to_parquet(DATA / "meta.parquet")


if __name__ == "__main__":
    main()
