"""Fine-tune mDeBERTa-v3-base as a sequence classifier (the strongest "standard" baseline)."""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import ROOT, SEED, eval_splits, load_split, save_preds, train_frame, truncate_words

MODEL = "microsoft/mdeberta-v3-base"


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(**enc).logits.float().softmax(-1).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux", action="store_true")
    ap.add_argument("--trunc", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()
    tag = f"trunc{args.trunc}" if args.trunc else "full"
    name = "mdeberta" + ("+aux" if args.aux else "")
    ckpt = ROOT / "models" / f"{name}__{tag}"
    torch.manual_seed(SEED)

    texts, labels = train_frame(args.aux, args.trunc)
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    tok = AutoTokenizer.from_pretrained(MODEL)
    if ckpt.exists():  # already trained: only re-predict (e.g. after new eval splits appear)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt).cuda()
        out = {s: predict(model, tok, truncate_words(load_split(s).text, args.trunc), args.max_len)
               for s in eval_splits(tag)}
        save_preds(name, tag, out, classes)
        return
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=len(classes)).cuda()

    def collate(idx):
        enc = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len, padding=True,
                  return_tensors="pt")
        enc["labels"] = torch.tensor(y[idx])
        return enc

    loader = DataLoader(np.arange(len(texts)), batch_size=args.bs, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(loader) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    for ep in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"epoch {ep}")
        for batch in pbar:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        val = load_split("val")
        p = predict(model, tok, truncate_words(val.text, args.trunc), args.max_len)
        print("val acc", (np.array(classes)[p.argmax(1)] == val.label.values).mean())

    out = {s: predict(model, tok, truncate_words(load_split(s).text, args.trunc), args.max_len)
           for s in eval_splits(tag)}
    save_preds(name, tag, out, classes)
    model.save_pretrained(ckpt)


if __name__ == "__main__":
    main()
