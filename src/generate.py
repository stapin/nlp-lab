"""Generate abstracts with new LLMs using the official AINL-Eval prompt.

  python generate.py --gen qwen2.5-3b          # one generator -> data/gen_qwen2.5-3b.parquet
  python generate.py --assemble                # build data/aux.parquet and data/heldout.parquet
"""
import argparse
import random
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import AUX_GENERATORS, DATA, HELDOUT_GENERATORS, OFFICIAL_PROMPT

OPENING_RE = re.compile(r"^В данной (статье|работе)\s+", re.I)
# Organisers "changed the beginnings with heuristics" (paper, Sec. 3.1); we mimic this so the
# opening phrase is not a trivial cue for our generators.
OPENINGS = ["В статье ", "В работе ", "В настоящей работе ", ""]
RNG = random.Random(42)
PREFIX_RE = re.compile(r"^\s*(Аннотация|Краткое содержание|Текст аннотации|Abstract)[^:\n]{0,40}:\s*", re.I)
MARKER_RE = re.compile(r"(Аннотация|Краткое содержание)[^:.\n]{0,40}[:.]\s*", re.I)


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.replace("**", "").replace("##", "").strip()
    # "<title>\n\nАннотация: ..." -> keep what follows the marker
    m = MARKER_RE.search(text[:300])
    if m:
        text = text[m.end():]
    text = PREFIX_RE.sub("", text)
    # a short first paragraph is a repeated title / heading
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) > 1 and len(paras[0].split()) < 15:
        paras = paras[1:]
    text = " ".join(paras)
    text = re.sub(r"\s*\n+\s*", " ", text)
    m = OPENING_RE.match(text)
    if m:
        rest, new = text[m.end():], RNG.choice(OPENINGS)
        text = new + rest if new else rest[:1].upper() + rest[1:]
    return re.sub(r"\s{2,}", " ", text).strip()


def generate(name, model_id, batch):
    meta = pd.read_parquet(DATA / "meta.parquet")
    meta = meta[meta.source == ("aux" if name in AUX_GENERATORS else "heldout")].reset_index(drop=True)

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    prompts = [
        tok.apply_chat_template(
            [{"role": "user", "content": OFFICIAL_PROMPT.format(title=r.title, keywords=r.keywords)}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        for r in meta.itertuples()
    ]
    outs = []
    for i in tqdm(range(0, len(prompts), batch), desc=name):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=320, do_sample=True, temperature=0.7, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    meta["raw"] = outs
    meta["text"] = [clean(o) for o in outs]
    meta["generator"] = name
    meta.to_parquet(DATA / f"gen_{name}.parquet")
    print(meta.text.str.split().str.len().describe())


def reclean():
    """Re-apply clean() to already generated files (after changing the post-processing)."""
    for f in DATA.glob("gen_*.parquet"):
        df = pd.read_parquet(f)
        df["text"] = [clean(o) for o in df.raw]
        df.to_parquet(f)


def assemble(min_words=20):
    def load(names):
        dfs = [pd.read_parquet(DATA / f"gen_{n}.parquet") for n in names if (DATA / f"gen_{n}.parquet").exists()]
        if not dfs:
            return pd.DataFrame(columns=["text", "generator"])
        df = pd.concat(dfs, ignore_index=True)
        return df[df.text.str.split().str.len() >= min_words]

    aux = load(AUX_GENERATORS)[["text", "generator"]].assign(label="unknown", split="aux")
    aux.reset_index(drop=True).to_parquet(DATA / "aux.parquet")
    print("aux", aux.generator.value_counts().to_dict())

    gen = load(HELDOUT_GENERATORS)[["text", "generator"]].assign(label="unknown")
    if len(gen):
        meta = pd.read_parquet(DATA / "meta.parquet")
        human = meta[meta.source == "heldout"][["text"]].assign(generator="human", label="human")
        held = pd.concat([human, gen], ignore_index=True).assign(split="heldout")
        held.to_parquet(DATA / "heldout.parquet")
        print("heldout", held.generator.value_counts().to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--reclean", action="store_true")
    args = ap.parse_args()
    if args.reclean:
        reclean()
    elif args.assemble:
        assemble()
    else:
        generate(args.gen, {**AUX_GENERATORS, **HELDOUT_GENERATORS}[args.gen], args.batch)
