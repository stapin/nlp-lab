"""LaTeX tables for the report, generated from results/metrics.csv -> report/tables/*.tex."""
import pandas as pd

from common import RESULTS, ROOT

OUT = ROOT / "report" / "tables"
M = pd.read_csv(RESULTS / "metrics.csv").set_index(["method", "tag"])

NAMES = {
    "tfidf": "TF-IDF + LR (official baseline)",
    "mdeberta": "mDeBERTa-v3-base, fine-tuned",
    "zeroshot-obs_logppl": "Log-perplexity, zero-shot",
    "zeroshot-binoculars": "Binoculars, zero-shot",
    "hybrid-lr[tfidf+lm+style]": "Hybrid: TF-IDF + LM + style (ours)",
    "hybrid-lr[tfidf+emb+lm+style]": "Hybrid: TF-IDF + emb + LM + style",
    "hybrid-lr[emb+lm+style]": "Hybrid: emb + LM + style",
}
MAIN = ["tfidf", "mdeberta", "hybrid-lr[emb+lm+style]", "hybrid-lr[tfidf+emb+lm+style]", "hybrid-lr[tfidf+lm+style]"]


def f(x, pct=False):
    if pd.isna(x):
        return "--"
    return f"{100 * x:.1f}" if pct else f"{x:.4f}"


def get(method, tag, col):
    try:
        return M.loc[(method, tag), col]
    except KeyError:
        return float("nan")


def bold_best(rows, cols):
    """rows: list of [name, v1, v2...] with float values; bold the max of each numeric column."""
    for j in cols:
        vals = [r[j] for r in rows if isinstance(r[j], float) and not pd.isna(r[j])]
        best = max(vals) if vals else None
        for r in rows:
            if isinstance(r[j], float):
                r[j] = "--" if pd.isna(r[j]) else (f"\\textbf{{{r[j]:.4f}}}" if r[j] == best else f"{r[j]:.4f}")
    return rows


def write(name, header, rows, align):
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    tex = f"\\begin{{tabular}}{{{align}}}\n\\hline\n{header} \\\\\n\\hline\n{body}\n\\hline\n\\end{{tabular}}\n"
    (OUT / f"{name}.tex").write_text(tex)


def main_table():
    rows = []
    for m in MAIN:
        for aux in ["", "+aux"]:
            if (m + aux, "full") not in M.index:
                continue
            rows.append([NAMES[m] + (" + pseudo-unknown" if aux else ""),
                         get(m + aux, "full", "dev_acc"), get(m + aux, "full", "dev_macro_f1"),
                         get(m + aux, "full", "dev_closed_acc"), get(m + aux, "full", "dev_unknown_recall")])
    write("main", "Method & Acc & Macro-F1 & Acc (4 known) & Recall$_{unknown}$",
          bold_best(rows, [1, 2, 3, 4]), "|l|cccc|")


def binary_table():
    rows = []
    for m in ["zeroshot-obs_logppl", "zeroshot-binoculars"] + MAIN:
        for aux in ["", "+aux"]:
            if (m + aux, "full") not in M.index:
                continue
            rows.append([NAMES[m] + (" + pseudo-unknown" if aux else ""),
                         get(m + aux, "full", "dev_bin_acc"), get(m + aux, "full", "dev_bin_f1"),
                         get(m + aux, "full", "dev_bin_auroc")])
    write("binary", "Method & Acc & F1 & ROC-AUC", bold_best(rows, [1, 2, 3]), "|l|ccc|")


def length_table():
    rows = []
    for m in ["zeroshot-binoculars", "tfidf", "mdeberta", "hybrid-lr[emb+lm+style]", "hybrid-lr[tfidf+lm+style]"]:
        for aux in ["", "+aux"]:
            if (m + aux, "trunc40") not in M.index:
                continue
            closed = [get(m + aux, t, "dev_closed_acc") for t in ["full", "trunc40"]]
            auc = [get(m + aux, t, "dev_bin_auroc") for t in ["full", "trunc40"]]
            rows.append([NAMES[m] + (" + pseudo-unknown" if aux else ""),
                         f(closed[0]), f(closed[1]),
                         "--" if pd.isna(closed[0]) else f"{100 * (closed[1] - closed[0]):+.1f}",
                         f(auc[0]), f(auc[1]), f"{100 * (auc[1] - auc[0]):+.1f}"])
    write("length", "Method & \\multicolumn{3}{c|}{Acc (4 known classes)} & \\multicolumn{3}{c|}{Human vs AI ROC-AUC} \\\\\n"
          " & full & 40 words & $\\Delta$, pp & full & 40 words & $\\Delta$, pp", rows, "|l|ccc|ccc|")


def ablation_table():
    groups = ["tfidf", "emb", "lm", "style", "lm+style", "emb+lm+style", "tfidf+lm", "tfidf+style", "tfidf+lm+style",
              "tfidf+emb+lm+style"]
    rows = []
    for g in groups:
        m = f"hybrid-lr[{g}]"
        rows.append([g.replace("+", " + ").replace("tfidf", "TF-IDF").replace("lm", "LM"),
                     get(m, "full", "dev_closed_acc"), get(m, "full", "dev_bin_auroc"),
                     get(m + "+aux", "full", "dev_acc"), get(m + "+aux", "full", "dev_unknown_recall")])
    rows.append(["emb + LM + style, LightGBM", get("hybrid-lgbm[emb+lm+style]", "full", "dev_closed_acc"),
                 get("hybrid-lgbm[emb+lm+style]", "full", "dev_bin_auroc"), float("nan"), float("nan")])
    write("ablation", "Feature groups & \\multicolumn{2}{c|}{4 known classes} & \\multicolumn{2}{c|}{+ pseudo-unknown} \\\\\n"
          " & Acc (4 known) & ROC-AUC & Acc (5 cl.) & Rec$_{unk}$", bold_best(rows, [1, 2, 3, 4]), "|l|cc|cc|")


def heldout_table():
    gens = sorted({c.split("[")[1].split("]")[0] for c in M.columns if c.startswith("heldout[")})
    if not gens:
        return
    rows = []
    for m in ["zeroshot-binoculars", "tfidf", "mdeberta", "hybrid-lr[tfidf+lm+style]"]:
        for aux in ["", "+aux"]:
            if (m + aux, "full") not in M.index or pd.isna(get(m + aux, "full", f"heldout[{gens[0]}]_auroc")):
                continue
            r = [NAMES[m] + (" + pseudo-unknown" if aux else "")]
            for g in gens:
                r += [get(m + aux, "full", f"heldout[{g}]_auroc"), get(m + aux, "full", f"heldout[{g}]_tpr@5fpr"),
                      get(m + aux, "full", f"heldout[{g}]_unknown_recall")]
            rows.append(r)
    header = "Method" + "".join(f" & \\multicolumn{{3}}{{c|}}{{{g}}}" for g in gens) + " \\\\\n" + \
             "".join(" & ROC-AUC & TPR@5\\% & Rec$_{unk}$" for _ in gens)
    write("heldout", header, bold_best(rows, list(range(1, 1 + 3 * len(gens)))), "|l|" + "ccc|" * len(gens))


def dataset_table():
    d = pd.read_csv(RESULTS / "dataset_stats.csv")
    rows, prev = [], None
    for r in d.itertuples():
        split = {"aux": "ours: pseudo-unknown", "heldout": "ours: held-out"}.get(r.split, r.split)
        if prev is not None and split != prev:
            rows.append(["\\hline %"])
        rows.append([split if split != prev else "", r._2, f"{r.texts:,}", f"{r.words_mean:.1f}",
                     f"{r.words_median:.0f}", f"{r.digits_per_100w:.1f}"])
        prev = split
    body = "\n".join((" & ".join(map(str, r)) + r" \\") if len(r) > 1 else "\\hline" for r in rows)
    tex = ("\\begin{tabular}{|l|l|rrrr|}\n\\hline\nSplit & Class / generator & Texts & Mean words & Median words & "
           "Digits / 100 w \\\\\n\\hline\n" + body + "\n\\hline\n\\end{tabular}\n")
    (OUT / "dataset.tex").write_text(tex)


def tex_escape(t):
    for a, b in [("\\", " "), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#"), ("$", "\\$"),
                 ("{", "("), ("}", ")"), ("~", "-"), ("^", "")]:
        t = t.replace(a, b)
    return t


def samples_table(pred_file="hybrid-lr[tfidf+lm+style]+aux__full", words=28, seed=3):
    """A few dev abstracts with the gold label and the prediction of the final model."""
    from common import load_split
    from evaluate import decide
    d = pd.read_parquet(RESULTS / "preds" / f"{pred_file}.parquet").query("split == 'dev'").reset_index(drop=True)
    d["text"] = load_split("dev").text.values
    d["pred"] = decide(d)
    n = d.text.str.split().str.len()
    picks = [
        d[(d.label == "human") & (d.pred == "human")].sample(1, random_state=seed),
        d[(d.label == "gpt-4-turbo") & (d.pred == "gpt-4-turbo")].sample(1, random_state=seed),
        d[(d.label == "unknown") & (d.pred == "unknown")].sample(1, random_state=seed),
        d[(d.label == "unknown") & (d.pred == "gemma-2-27b")].sample(1, random_state=seed),
        d[(d.label == "human") & (d.pred != "human") & (n < 60)].sample(1, random_state=seed),
    ]
    rows = []
    for r in pd.concat(picks).itertuples():
        text = tex_escape(" ".join(r.text.split()[:words])) + (" \\dots" if len(r.text.split()) > words else "")
        rows.append(f"\\parbox[t]{{0.62\\linewidth}}{{\\raggedright\\small {text}}} & {r.label} & {r.pred} \\\\ \\hline")
    tex = ("\\begin{tabular}{|p{0.64\\linewidth}|l|l|}\n\\hline\nAbstract (first " + str(words) +
           " words) & Gold & Predicted \\\\\n\\hline\n" + "\n".join(rows) + "\n\\end{tabular}\n")
    (OUT / "samples.tex").write_text(tex)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    samples_table()
    dataset_table()
    main_table()
    binary_table()
    length_table()
    ablation_table()
    heldout_table()
    print(*sorted(p.name for p in OUT.iterdir()))
